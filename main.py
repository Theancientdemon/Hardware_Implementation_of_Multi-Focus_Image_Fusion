import os.path
import sys
from datetime import datetime
from enum import Enum, auto

import cv2
import pygame
import colors
from Tools import IllegalEntryError
from algorithms.Fusion import Fusion
from algorithms.Registration import Registration


class State(Enum):
    CAMERA = auto()
    QUICK = auto()
    SETTINGS = auto()
    ADVANCE_SETTING = auto()
    ASK_PHOTO = auto()
    ASK_FUSE = auto()
    ASK_QUIT = auto()
    VIEW_PHOTO = auto()

    LEVEL_SELECT = auto()
    WAVE_SELECT = auto()
    RULE_SELECT = auto()
    CHANNEL_SELECT = auto()
    ADVANCE_RULE_SETTING = auto()

class App:
    def __init__(self):
        self.settings_file_path = "settings_do_not_open.txt"
        # Wavelet List
        self.wavelet_list = ["haar", "db1", "db2", "db3", "db4", "db5", "db6", "db7", "db8", "db9", "db10"]
        # Rules List
        self.rule_list = ["max", "min", "avg", "lv", "sml"]

        pygame.init()
        self.focusValue = 0
        self.screensize = (480,320)

        if "-t" in sys.argv:
            self.testcase = True
        else:
            self.testcase = False
            from picamera2 import Picamera2
            # Camera Config
            self.cam = Picamera2()
            config = self.cam.create_still_configuration(main={"size": (480, 320)}, lores={"size": (480, 320)})
            self.cam.configure(config)
            self.cam.start()

        if "-f" in sys.argv:
            self.screen = pygame.display.set_mode(self.screensize, pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode(self.screensize)

        self.running = True
        self.clock = pygame.time.Clock()

        # Input Type for which input is taken
        # Keyboard
        # Buttons for actual push buttons
        # Touch for Touch screen controls
        # Joy for Joystick controller
        self.inputType = "Keyboard"
        if self.inputType == "Joy":
            pygame.joystick.init()
            if pygame.joystick.get_count() == 0:
                # no Joystic connected
                # fallback to keyboard
                self.inputType = "Keyboard"

        # State defaults to and starts at CAMERA MODE
        self.state = State.CAMERA
        self.viewPhotoSurface = None
        self.capturedIMG_path = None
        self.img1_path = None
        self.img2_path = None
        self.fused_path = None

        # State Specific variables
        # ------------------------------------ #
        self.quick_active = 0
        self.settings_active = 0
        self.advance_setting_active = 0
        self.wave_active = 0
        self.level_active = 0
        self.rule_active = 0
        self.advance_rule_setting_active = 0
        self.channel_sel_active = 0
        # ------------------------------------ #

        self.loadSettings()
        self.createAssets()
        self.appLoop()

    def quit(self):
        pygame.quit()

    def appLoop(self):
        while self.running:
            self.screen.fill(colors.BLACK)
            self.inputHandler()

            # noinspection PyUnreachableCode
            match self.state:
                case State.CAMERA:
                    self.renderCamera()
                case State.QUICK:
                    self.renderQuick()
                case State.SETTINGS:
                    self.renderSettings()
                case State.ASK_QUIT:
                    self.askQuit()
                case State.ADVANCE_SETTING:
                    self.renderAdvanceSetting()
                case State.LEVEL_SELECT:
                    self.renderLevelSel()
                case State.WAVE_SELECT:
                    self.renderWaveSel()
                case State.RULE_SELECT:
                    self.renderRuleSel()
                case State.ADVANCE_RULE_SETTING:
                    self.renderAdvanceRuleSetting()
                case State.CHANNEL_SELECT:
                    self.renderChannelSelect()
                case State.ASK_FUSE:
                    self.renderAskFuse()
                case State.ASK_PHOTO:
                    self.renderAskPhoto()
                case State.VIEW_PHOTO:
                    self.renderViewPhoto()
                case _:
                    # Fall back to CAMERA MODE
                    self.state = State.CAMERA

            # Update the screen
            pygame.display.flip()
            self.clock.tick(30)
        self.quit()

    def inputHandler(self):
        match self.inputType:
            case "Keyboard":
                self.keyboardHandler()
            case "Joy":
                self.joystickHandler()
            case "Buttons":
                self.buttonHandler()
            case "Touch":
                self.touchHandler()
            case _:
                # Default to Keyboard
                self.keyboardHandler()

    def keyboardHandler(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.JOYBUTTONDOWN:
                self.inputType = "Joy"
            elif event.type == pygame.KEYDOWN:
                match event.key:
                    case pygame.K_ESCAPE:
                        self.running = False
                    case pygame.K_a:
                        self.B_key()
                    case pygame.K_s:
                        self.down_key()
                    case pygame.K_w:
                        self.up_key()
                    case pygame.K_d:
                        self.A_key()

    def joystickHandler(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.JOYDEVICEREMOVED:
                if pygame.joystick.get_count() == 0:
                    # fallback to keyboard
                    self.inputType = "Keyboard"
            elif event.type == pygame.KEYDOWN:
                self.inputType = "keyboard"
            elif event.type == pygame.JOYBUTTONDOWN:
                match event.button:
                    case 0:
                        self.down_key()
                    case 1:
                        self.A_key()
                    case 2:
                        self.B_key()
                    case 3:
                        self.up_key()

    def buttonHandler(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def touchHandler(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def B_key(self):
        match self.state:
            case State.CAMERA:
                self.state = State.QUICK
            case State.QUICK:
                self.state = State.CAMERA
            case State.SETTINGS:
                self.state = State.QUICK
            case State.ASK_QUIT:
                self.state = State.QUICK
            case State.ADVANCE_SETTING:
                self.state = State.SETTINGS
            case State.WAVE_SELECT:
                self.state = State.ADVANCE_SETTING
            case State.LEVEL_SELECT:
                self.state = State.ADVANCE_SETTING
            case State.RULE_SELECT:
                self.state = State.ADVANCE_RULE_SETTING
            case State.ADVANCE_RULE_SETTING:
                self.state = State.SETTINGS
            case State.ASK_FUSE:
                self.state = State.CAMERA
                self.img1_path = None
                self.img2_path = None
            case State.VIEW_PHOTO:
                self.state = State.ASK_PHOTO
                self.viewPhotoSurface = None
            case State.ASK_PHOTO:
                self.state = State.CAMERA
            case State.CHANNEL_SELECT:
                self.state = State.ADVANCE_RULE_SETTING

    def down_key(self):
        match self.state:
            case State.CAMERA:
                self.focusNear()
            case State.QUICK:
                self.quick_active += 1
                self.quick_active %= 4
            case State.SETTINGS:
                self.settings_active += 1
                self.settings_active %= 3
            case State.ADVANCE_SETTING:
                self.advance_setting_active += 1
                self.advance_setting_active %= 2
            case State.WAVE_SELECT:
                self.wave_active += 1
                self.wave_active %= len(self.wavelet_list)
            case State.LEVEL_SELECT:
                self.level_active += 1
                self.level_active %= 10
            case State.RULE_SELECT:
                self.rule_active += 1
                self.rule_active %= len(self.rule_list)
            case State.ADVANCE_RULE_SETTING:
                self.advance_rule_setting_active += 1
                self.advance_rule_setting_active %= 3
            case State.CHANNEL_SELECT:
                self.channel_sel_active += 1
                self.channel_sel_active %= 2
            case State.VIEW_PHOTO:
                self.state = State.ASK_PHOTO
                self.viewPhotoSurface = None

    def up_key(self):
        match self.state:
            case State.CAMERA:
                self.focusFar()
            case State.QUICK:
                self.quick_active -= 1
                self.quick_active %= 4
            case State.SETTINGS:
                self.settings_active -= 1
                self.settings_active %= 3
            case State.ADVANCE_SETTING:
                self.advance_setting_active -= 1
                self.advance_setting_active %= 2
            case State.WAVE_SELECT:
                self.wave_active -= 1
                self.wave_active %= len(self.wavelet_list)
            case State.LEVEL_SELECT:
                self.level_active -= 1
                self.level_active %= 10
            case State.RULE_SELECT:
                self.rule_active -= 1
                self.rule_active %= len(self.rule_list)
            case State.ADVANCE_RULE_SETTING:
                self.advance_rule_setting_active -= 1
                self.advance_rule_setting_active %= 3
            case State.CHANNEL_SELECT:
                self.channel_sel_active += 1
                self.channel_sel_active %= 2
            case State.VIEW_PHOTO:
                self.state = State.ASK_PHOTO
                self.viewPhotoSurface = None
            case State.ASK_PHOTO:
                self.state = State.VIEW_PHOTO

    def A_key(self):
        match self.state:
            case State.CAMERA:
                self.capturePhoto()
            case State.QUICK:
                match self.quick_active:
                    case 0:
                        self.state = State.SETTINGS
                    case 1:
                        self.do_Fusion = not self.do_Fusion
                    case 2:
                        self.do_registration = not self.do_registration
                    case 3:
                        self.state = State.ASK_QUIT
                    case _:
                        self.quick_active = 0
            case State.SETTINGS:
                if self.settings_active == 2:
                    self.state = State.ADVANCE_RULE_SETTING
                else:
                    self.state = State.ADVANCE_SETTING
            case State.ASK_QUIT:
                self.running = False
            case State.ADVANCE_SETTING:
                match self.advance_setting_active:
                    case 0:
                        self.state = State.WAVE_SELECT
                    case 1:
                        self.state = State.LEVEL_SELECT
                    case _:
                        self.advance_setting_active = 0
            case State.WAVE_SELECT:
                if self.settings_active == 0:
                    self.fusion_wavelet = self.wavelet_list[self.wave_active]
                elif self.settings_active == 1:
                    self.registration_wavelet = self.wavelet_list[self.wave_active]
                self.state = State.ADVANCE_SETTING
                self.saveSettings()
            case State.LEVEL_SELECT:
                if self.settings_active == 0:
                    self.fusion_level = self.level_active + 1
                elif self.settings_active == 1:
                    self.registration_level = self.level_active + 1
                self.state = State.ADVANCE_SETTING
                self.saveSettings()
            case State.RULE_SELECT:
                if self.advance_rule_setting_active:
                    self.detail_rule = self.rule_list[self.rule_active]
                else:
                    self.approx_rule = self.rule_list[self.rule_active]
                self.state = State.ADVANCE_RULE_SETTING
                self.saveSettings()
            case State.ADVANCE_RULE_SETTING:
                if self.advance_rule_setting_active == 2:
                    self.state = State.CHANNEL_SELECT
                else:
                    self.state = State.RULE_SELECT
            case State.CHANNEL_SELECT:
                if self.channel_sel_active:
                    self.fuse_channel = 3
                else:
                    self.fuse_channel = 1
                self.state = State.ADVANCE_RULE_SETTING
                self.saveSettings()
            case State.VIEW_PHOTO:
                self.state = State.ASK_PHOTO
                self.viewPhotoSurface = None
            case State.ASK_PHOTO:
                if self.img1_path is None:
                    self.img1_path = self.capturedIMG_path
                    self.state = State.CAMERA
                else:
                    self.img2_path = self.capturedIMG_path
                    self.state = State.ASK_FUSE
            case State.ASK_FUSE:
                self.fuse_photos()
                self.state = State.CAMERA

    def createAssets(self):
        small_symbol_font = pygame.font.SysFont("Arial", 20)
        self.big_symbol_font = pygame.font.SysFont("Arial", 50)

        self.screenPlaceHolder = pygame.image.load("assets/img_2.png")
        self.settingsIcon = pygame.image.load("assets/settingsicon2.png")
        self.powerIcon = pygame.image.load("assets/power_button.png")

        self.reg_symbol_small = pygame.Surface((20,20))
        self.reg_symbol_small.fill(colors.BLACK)
        temp = small_symbol_font.render("R",True, colors.WHITE)
        self.reg_symbol_small.blit(temp, (5,0), temp.get_rect())

        self.fuse_symbol_small = pygame.Surface((20,20))
        self.fuse_symbol_small.fill(colors.BLACK)
        temp = small_symbol_font.render("F",True, colors.WHITE)
        self.fuse_symbol_small.blit(temp, (5,0), temp.get_rect())

        self.star_symbol_small = pygame.Surface((20,20))
        self.star_symbol_small.fill(colors.BLACK)
        temp = small_symbol_font.render("*",True, colors.WHITE)
        self.star_symbol_small.blit(temp, (5,0), temp.get_rect())

        self.reg_symbol_big_off = pygame.Surface((50, 50))
        self.reg_symbol_big_off.fill(colors.BLACK)
        temp = self.big_symbol_font.render("R", True, colors.WHITE)
        self.reg_symbol_big_off.blit(temp, (10, 0), temp.get_rect())

        self.fuse_symbol_big_off = pygame.Surface((50, 50))
        self.fuse_symbol_big_off.fill(colors.BLACK)
        temp = self.big_symbol_font.render("F", True, colors.WHITE)
        self.fuse_symbol_big_off.blit(temp, (10, 0), temp.get_rect())

        self.reg_symbol_big_on = pygame.Surface((50, 50))
        self.reg_symbol_big_on.fill(colors.DARK_GREEN)
        temp = self.big_symbol_font.render("R", True, colors.BLACK)
        self.reg_symbol_big_on.blit(temp, (10, 0), temp.get_rect())

        self.fuse_symbol_big_on = pygame.Surface((50, 50))
        self.fuse_symbol_big_on.fill(colors.DARK_GREEN)
        temp = self.big_symbol_font.render("F", True, colors.BLACK)
        self.fuse_symbol_big_on.blit(temp, (10, 0), temp.get_rect())

        self.please_wait_screen = pygame.Surface(self.screensize)
        self.please_wait_screen.fill(colors.BLACK)
        temp = self.big_symbol_font.render("PLEASE WAIT", True, colors.WHITE)
        self.please_wait_screen.blit(temp, (0, 0), temp.get_rect())

        self.fusion_text = self.big_symbol_font.render("FUSION", True, colors.WHITE)
        self.registration_text = self.big_symbol_font.render("REGISTRATION", True, colors.WHITE)
        self.settings_text = self.big_symbol_font.render("SETTINGS", True, colors.WHITE)
        self.wavelet_text = self.big_symbol_font.render("WAVELET", True, colors.WHITE)
        self.level_text = self.big_symbol_font.render("LEVEL", True, colors.WHITE)
        self.quit_text = self.big_symbol_font.render("QUIT..", True, colors.WHITE)
        self.rule_text = self.big_symbol_font.render("RULE", True, colors.WHITE)
        self.approx_text = self.big_symbol_font.render("APPROX.", True, colors.WHITE)
        self.detail_text = self.big_symbol_font.render("DETAIL", True, colors.WHITE)
        self.channel_text = self.big_symbol_font.render("CHANNEL", True, colors.WHITE)
        self.askFuse_text = self.big_symbol_font.render("FUSE PHOTOS", True, colors.WHITE)
        self.askPhotos_text = self.big_symbol_font.render("USE FOR FUSE", True, colors.WHITE)
        self.view_text = small_symbol_font.render("VIEW", True, colors.WHITE)
        self.up_text = self.big_symbol_font.render("^", True, colors.WHITE)

        self.yes_text = pygame.Surface((100,50))
        self.yes_text.fill(colors.GREEN)
        temp = self.big_symbol_font.render("YES", True, colors.BLACK)
        self.yes_text.blit(temp, (10, 0), temp.get_rect())

        self.no_text = pygame.Surface((100,50))
        self.no_text.fill(colors.RED)
        temp = self.big_symbol_font.render("NO", True, colors.BLACK)
        self.no_text.blit(temp, (20, 0), temp.get_rect())

    def renderCamera(self):
        if self.testcase:
            temp = self.screenPlaceHolder
        else:
            temp = self.cam.capture_array("lores")
            temp = cv2.cvtColor(temp, cv2.COLOR_YUV420p2RGB)
            temp = pygame.surfarray.make_surface(temp.swapaxes(0,1))
        self.screen.blit(temp, (0, 0), temp.get_rect())

        if self.img1_path is not None:
            self.screen.blit(self.star_symbol_small, (10, 230), self.reg_symbol_small.get_rect())

        if self.do_registration:
            self.screen.blit(self.reg_symbol_small, (10, 290), self.reg_symbol_small.get_rect())

        if self.do_Fusion:
            self.screen.blit(self.fuse_symbol_small, (10, 260), self.fuse_symbol_small.get_rect())

    def renderQuick(self):
        self.renderCamera()
        quicksettingsurface = pygame.Surface((100,360))
        quicksettingsurface.fill(colors.BLACK)

        if self.do_registration:
            quicksettingsurface.blit(self.reg_symbol_big_on,(25,175), self.reg_symbol_big_on.get_rect())
        else:
            quicksettingsurface.blit(self.reg_symbol_big_off,(25,175), self.reg_symbol_big_off.get_rect())
        if self.do_Fusion:
            quicksettingsurface.blit(self.fuse_symbol_big_on,(25,95), self.fuse_symbol_big_on.get_rect())
        else:
            quicksettingsurface.blit(self.fuse_symbol_big_off,(25,95), self.fuse_symbol_big_off.get_rect())

        quicksettingsurface.blit(self.settingsIcon,(25,15), self.settingsIcon.get_rect())
        quicksettingsurface.blit(self.powerIcon, (25, 255), self.powerIcon.get_rect())
        pygame.draw.rect(quicksettingsurface, colors.WHITE, (25, 15+(self.quick_active*80),50, 50), 4)

        self.screen.blit(quicksettingsurface, (380,0), quicksettingsurface.get_rect())

    def renderSettings(self):
        self.screen.blit(self.settings_text, (10,0), self.settings_text.get_rect())
        self.screen.blit(self.fusion_text, (10,50), self.fusion_text.get_rect())
        self.screen.blit(self.registration_text, (10,100), self.registration_text.get_rect())
        self.screen.blit(self.rule_text, (10,150), self.registration_text.get_rect())

        pygame.draw.rect(self.screen, colors.WHITE, (0, 50+50*self.settings_active, 480, 50), 5)

    def askQuit(self):
        self.screen.blit(self.quit_text, (180, 100), self.quit_text.get_rect())
        self.screen.blit(self.yes_text, (310, 200), self.yes_text.get_rect())
        self.screen.blit(self.no_text, (70, 200), self.no_text.get_rect())

    def renderAskFuse(self):
        self.screen.blit(self.askFuse_text, (90, 100), self.askFuse_text.get_rect())
        self.screen.blit(self.yes_text, (310, 200), self.yes_text.get_rect())
        self.screen.blit(self.no_text, (70, 200), self.no_text.get_rect())

    def renderAskPhoto(self):
        self.screen.blit(self.askPhotos_text, (90, 100), self.askPhotos_text.get_rect())
        self.screen.blit(self.yes_text, (310, 200), self.yes_text.get_rect())
        self.screen.blit(self.no_text, (70, 200), self.no_text.get_rect())
        self.screen.blit(self.view_text, (220, 200), self.view_text.get_rect())
        self.screen.blit(self.up_text, (230, 220), self.up_text.get_rect())

    def renderViewPhoto(self):
        if self.viewPhotoSurface is None:
            temp = cv2.imread(self.capturedIMG_path)
            temp = cv2.resize(temp, self.screensize)
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
            self.viewPhotoSurface = pygame.surfarray.make_surface(temp.swapaxes(0,1))
        self.screen.blit(self.viewPhotoSurface, (0,0), self.viewPhotoSurface.get_rect())

    def renderAdvanceSetting(self):
        if self.settings_active == 0:
            self.screen.blit(self.fusion_text, (10, 0), self.fusion_text.get_rect())
            wave_txt = self.big_symbol_font.render(self.fusion_wavelet, True, colors.GREEN)
            lvl_txt = self.big_symbol_font.render(str(self.fusion_level), True, colors.GREEN)
        elif self.settings_active == 1:
            self.screen.blit(self.registration_text, (10, 0), self.registration_text.get_rect())
            wave_txt = self.big_symbol_font.render(self.registration_wavelet, True, colors.GREEN)
            lvl_txt = self.big_symbol_font.render(str(self.registration_level), True, colors.GREEN)
        else:
            raise IllegalEntryError("illegal entry into advance settings")
        self.screen.blit(self.wavelet_text, (10, 50), self.wavelet_text.get_rect())
        self.screen.blit(self.level_text, (10, 100), self.level_text.get_rect())
        self.screen.blit(wave_txt, (250, 50), wave_txt.get_rect())
        self.screen.blit(lvl_txt, (250, 100), lvl_txt.get_rect())

        pygame.draw.rect(self.screen, colors.WHITE, (0, 50+50*self.advance_setting_active, 480, 50), 5)

    def renderWaveSel(self):
        active_top = self.wave_active * 60
        active_bottom = active_top + 60
        scroll_offset = 0
        if active_top < scroll_offset:
            scroll_offset = active_top
        elif active_bottom > scroll_offset + 320:
            scroll_offset = active_bottom - 320

        for i, wave in enumerate(self.wavelet_list):
            y = i * 60 - scroll_offset
            if y < -60 or y > 320:
                continue
            if self.settings_active == 1:
                temp = self.big_symbol_font.render(
                    wave,
                    True,
                    colors.GREEN if self.registration_wavelet == wave else colors.WHITE)
            elif self.settings_active == 0:
                temp = self.big_symbol_font.render(
                    wave,
                    True,
                    colors.GREEN if self.fusion_wavelet == wave else colors.WHITE)
            else:
                raise IllegalEntryError("illegal entry into wavelet select screen")
            self.screen.blit(temp,(10,y), temp.get_rect())
        pygame.draw.rect(self.screen, colors.WHITE, (0, 60*self.wave_active - scroll_offset, 480, 60), 5)

    def renderLevelSel(self):
        active_top = self.level_active * 60
        active_bottom = active_top + 60
        scroll_offset = 0
        if active_top < scroll_offset:
            scroll_offset = active_top
        elif active_bottom > scroll_offset + 320:
            scroll_offset = active_bottom - 320

        for i in range(10):
            y = i * 60 - scroll_offset
            if y < -60 or y > 320:
                continue
            if self.settings_active == 1:
                temp = self.big_symbol_font.render(
                    str(i + 1),
                    True,
                    colors.GREEN if self.registration_level == i+1 else colors.WHITE)
            elif self.settings_active == 0:
                temp = self.big_symbol_font.render(
                    str(i + 1),
                    True,
                    colors.GREEN if self.fusion_level == i+1 else colors.WHITE)
            else:
                raise IllegalEntryError("illegal entry into level select screen")
            self.screen.blit(temp, (10, y), temp.get_rect())
        pygame.draw.rect(self.screen, colors.WHITE, (0, 60 * self.level_active - scroll_offset, 480, 60), 5)

    def renderRuleSel(self):
        active_top = self.rule_active * 60
        active_bottom = active_top + 60
        scroll_offset = 0
        if active_top < scroll_offset:
            scroll_offset = active_top
        elif active_bottom > scroll_offset + 320:
            scroll_offset = active_bottom - 320

        for i, rule in enumerate(self.rule_list):
            y = i * 60 - scroll_offset
            if y < -60 or y > 320:
                continue
            if self.advance_rule_setting_active:
                temp = self.big_symbol_font.render(
                    rule,
                    True,
                    colors.GREEN if self.detail_rule == rule else colors.WHITE)
            else:
                temp = self.big_symbol_font.render(
                    rule,
                    True,
                    colors.GREEN if self.approx_rule == rule else colors.WHITE)
            self.screen.blit(temp, (10, y), temp.get_rect())
        pygame.draw.rect(self.screen, colors.WHITE, (0, 60 * self.rule_active - scroll_offset, 480, 60), 5)

    def renderAdvanceRuleSetting(self):
        self.screen.blit(self.rule_text, (10, 0), self.rule_text.get_rect())
        self.screen.blit(self.approx_text, (10, 50), self.approx_text.get_rect())
        temp = self.big_symbol_font.render(self.approx_rule, True, colors.GREEN)
        self.screen.blit(temp, (250, 50), temp.get_rect())
        self.screen.blit(self.detail_text, (10, 100), self.detail_text.get_rect())
        temp = self.big_symbol_font.render(self.detail_rule, True, colors.GREEN)
        self.screen.blit(temp, (250, 100), temp.get_rect())
        self.screen.blit(self.channel_text, (10, 150), self.channel_text.get_rect())
        temp = self.big_symbol_font.render(str(self.fuse_channel), True, colors.GREEN)
        self.screen.blit(temp, (250, 150), temp.get_rect())

        pygame.draw.rect(self.screen, colors.WHITE, (0, 50 + 50 * self.advance_rule_setting_active, 480, 50), 5)

    def renderChannelSelect(self):
        self.screen.blit(self.rule_text, (10, 0), self.rule_text.get_rect())
        temp = self.big_symbol_font.render("1", True, colors.GREEN)
        self.screen.blit(temp, (10, 50), temp.get_rect())
        temp = self.big_symbol_font.render("3", True, colors.GREEN)
        self.screen.blit(temp, (10, 100), temp.get_rect())

        pygame.draw.rect(self.screen, colors.WHITE, (0, 50 + 50 * self.channel_sel_active, 480, 50), 5)

    def capturePhoto(self):
        print("capturing image")
        if self.testcase:
            if self.img1_path is None:
                self.capturedIMG_path = "photos/test/book1.jpg"
            else:
                self.capturedIMG_path = "photos/test/book2.jpg"
        else:
            now = datetime.now()
            formatted = f"photos/captured/{now.year}{now.month:02}{now.day:02}{now.hour:02}{now.minute:02}{now.second:02}.png"
            print(formatted)
            self.capturedIMG_path = formatted
            self.cam.capture_file(self.capturedIMG_path)
        print(f"file saved, {self.capturedIMG_path}")
        if self.do_Fusion:
            self.state = State.ASK_PHOTO

    def focusFar(self):
        #command v4l2-ctl -d /dev/v4l-subdev1 -c focus_absolute={value} 0-1023
        self.focusValue -= 50
        if self.focusValue < 0:
            self.focusValue = 0
        os.system(f"v4l2-ctl -d /dev/v4l-subdev1 -c focus_absolute={self.focusValue}")
        print(f"focus:{self.focusValue}")

    def focusNear(self):
        self.focusValue += 50
        if self.focusValue > 1000:
            self.focusValue = 1000
        os.system(f"v4l2-ctl -d /dev/v4l-subdev1 -c focus_absolute={self.focusValue}")
        print(f"focus:{self.focusValue}")

    def loadSettings(self):
        if os.path.exists(self.settings_file_path):
            with open(self.settings_file_path) as file: data = file.readlines()
            self.do_Fusion = True if data[0].strip() == "True" else False
            self.do_registration = True if data[1].strip() == "True" else False
            self.fusion_wavelet = data[2].strip() if data[2].strip() in self.wavelet_list else self.wavelet_list[0]
            self.registration_wavelet = data[3].strip() if data[3].strip() in self.wavelet_list else self.wavelet_list[0]
            self.fusion_level = int(data[4]) if data[4].strip().isdigit() else 2
            self.registration_level = int(data[5]) if data[5].strip().isdigit() else 2
            self.approx_rule = data[6].strip() if data[6].strip() in self.rule_list else self.rule_list[0]
            self.detail_rule = data[7].strip() if data[7].strip() in self.rule_list else self.rule_list[0]
            self.fuse_channel = int(data[8]) if data[8].strip() in ("1", "3") else 3
        else:
            # Default settings
            self.do_Fusion = True
            self.do_registration = True
            self.fusion_wavelet = self.wavelet_list[0]
            self.registration_wavelet = self.wavelet_list[0]
            self.fusion_level = 2
            self.registration_level = 2
            self.approx_rule = self.rule_list[0]
            self.detail_rule = self.rule_list[0]
            self.fuse_channel = 3

    def saveSettings(self):
        with open(self.settings_file_path, "wt") as file:
            file.write(str(self.do_Fusion)+"\n")
            file.write(str(self.do_registration)+"\n")
            file.write(self.fusion_wavelet+"\n")
            file.write(self.registration_wavelet+"\n")
            file.write(str(self.fusion_level)+"\n")
            file.write(str(self.registration_level)+"\n")
            file.write(self.approx_rule+"\n")
            file.write(self.detail_rule+"\n")
            file.write(str(self.fuse_channel)+"\n")

    def fuse_photos(self):
        if self.do_registration:
            self.img2_path = Registration.register(self.img1_path,
                                                   self.img2_path,
                                                   self.registration_wavelet,
                                                   self.registration_level)
        self.fused_path = Fusion.fuse(self.img1_path,
                                      self.img2_path,
                                      self.fusion_wavelet,
                                      self.fusion_level,
                                      self.approx_rule,
                                      self.detail_rule,
                                      self.fuse_channel)
        self.img1_path = None
        self.img2_path = None


if __name__ == "__main__":
    App()