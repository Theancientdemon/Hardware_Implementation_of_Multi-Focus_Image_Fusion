import os.path
from argparse import ArgumentParser
from datetime import datetime
from enum import Enum, auto

import cv2
import pygame

import colors
from Tools import SettingsError
from algorithms.Fusion import Fusion
from algorithms.Registration import Registration


class State(Enum):
    """
        Dictates which State the app is on (Which screen is app on).
    """
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
    """
        Main Class for Whole app.
        Contains UI as well as controls.
        USAGE: App()
        Note: All process is done in constructor.
    """
    def __init__(self):
        # Settings file.
        # Don't manually change file but through app only.
        self.settings_file_path = "settings_do_not_open.txt"
        # To update the wavelet list and rules list. Make changes to "algorithms/Fusion.py"
        # Wavelet List
        self.wavelet_list = Fusion.Wavelet_list
        # Rules List
        self.rule_list = Fusion.Rules

        self.parse_args()

        pygame.init()
        self.focusValue = 0
        # The app is designed for below size only. changing it may cause issues
        self.screensize = (480,320)

        if self.args.test:
            # Can work on non-raspberry pi devices.
            self.testcase = True
        else:
            # Only works on raspberry pi due to dependencies
            self.testcase = False
            from picamera2 import Picamera2  # rpi only dependencys
            # Camera Config
            self.cam = Picamera2()
            config = self.cam.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (480, 320)})
            self.cam.configure(config)
            self.cam.start()

        if self.args.fullscreen:
            self.screen = pygame.display.set_mode(self.screensize, pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode(self.screensize)

        self.send_data = self.args.send is not None
        self.device = self.args.send
        self.running = True

        # Input Type for which input is taken
        # Keyboard
        # Buttons for actual push buttons
        # Touch for Touch screen controls
        # Joy for Joystick controller
        self.inputType = "Joy"
        if self.inputType == "Joy":
            pygame.joystick.init()
            if pygame.joystick.get_count() == 0:
                # no Joystic connected
                # fallback to keyboard
                self.inputType = "Keyboard"
        print(f"{self.inputType=}")

        # State defaults to and starts at CAMERA MODE
        self.state = State.CAMERA
        self.viewPhotoSurface = None
        self.capturedIMG_path = None
        self.capturedIMG_focus = 0
        self.img1_path = None
        self.img1_focus = 0
        self.img2_path = None
        self.img2_focus = 0
        self.fused_path = None
        self.registered_path = None
        self.did_fusion = False #used to show fused images after fusion

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
        """Closes the app"""
        pygame.quit()

    def appLoop(self):
        """Main loop which keeps the app running"""
        while self.running:
            self.screen.fill(colors.BLACK)
            self.inputHandler()

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
        self.quit()

    def inputHandler(self):
        """
        Handles the input of app.
        Can be changed by changing var "inputType".
        """
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
                # fallback to keyboard
                print(f"\033[93mWARNING!:\033[0m '{self.inputType}' is not a valid input type. Falling back to Keyboard.")
                self.inputType = "Keyboard"

    def keyboardHandler(self):
        """
        Handles Input from a Keyboard connected to the device
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.JOYBUTTONDOWN:
                self.inputType = "Joy"
                print(f"{self.inputType=}")
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
        """
        Handles input from a Video Game Controller.
        """
        for event in pygame.event.get():
            # print(event)
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.JOYDEVICEREMOVED:
                if pygame.joystick.get_count() == 0:
                    # fallback to keyboard
                    self.inputType = "Keyboard"
                print(f"{self.inputType=}")
            elif event.type == pygame.KEYDOWN:
                self.inputType = "keyboard"
                print(f"{self.inputType=}")
            elif event.type == pygame.JOYBUTTONDOWN:
                # print(f"button:{event.button}")
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
        """
        Handles input from the mechanical buttons connected to GPIO.
        """
        raise NotImplementedError("Buttons is to be implemented. program doesn't support buttons inputs yet")

    def touchHandler(self):
        """
        Handles input from the touch screen.
        """
        raise NotImplementedError("touch input is to be implemented. program doesn't support buttons inputs yet")

    def B_key(self):
        """
        Actions to be done when the 'B' Key is pressed.
        """
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
                if self.did_fusion:
                    self.state = State.CAMERA
                else:
                    self.state = State.ASK_PHOTO
                self.viewPhotoSurface = None
            case State.ASK_PHOTO:
                self.state = State.CAMERA
            case State.CHANNEL_SELECT:
                self.state = State.ADVANCE_SETTING

    def down_key(self):
        """
        Actions to be done when the 'down' Key is pressed.
        """
        match self.state:
            case State.CAMERA:
                self.focusNear()
            case State.QUICK:
                self.quick_active += 1
                self.quick_active %= 4
            case State.SETTINGS:
                self.settings_active += 1
                self.settings_active %= 2
            case State.ADVANCE_SETTING:
                self.advance_setting_active += 1
                self.advance_setting_active %= 3
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
                self.advance_rule_setting_active %= 2
            case State.CHANNEL_SELECT:
                self.channel_sel_active += 1
                self.channel_sel_active %= 2
            case State.VIEW_PHOTO:
                if self.did_fusion:
                    self.state = State.CAMERA
                else:
                    self.state = State.ASK_PHOTO
                self.viewPhotoSurface = None

    def up_key(self):
        """
        Actions to be done when the 'up' Key is pressed.
        """
        match self.state:
            case State.CAMERA:
                self.focusFar()
            case State.QUICK:
                self.quick_active -= 1
                self.quick_active %= 4
            case State.SETTINGS:
                self.settings_active -= 1
                self.settings_active %= 2
            case State.ADVANCE_SETTING:
                self.advance_setting_active -= 1
                self.advance_setting_active %= 3
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
                self.advance_rule_setting_active %= 2
            case State.CHANNEL_SELECT:
                self.channel_sel_active += 1
                self.channel_sel_active %= 2
            case State.VIEW_PHOTO:
                if self.did_fusion:
                    self.state = State.CAMERA
                else:
                    self.state = State.ASK_PHOTO
                self.viewPhotoSurface = None
            case State.ASK_PHOTO:
                self.state = State.VIEW_PHOTO

    def A_key(self):
        """
        Actions to be done when the 'A' Key is pressed.
        """
        match self.state:
            case State.CAMERA:
                self.capturePhoto()
                if self.send_data: self.sendPhoto(self.capturedIMG_path)
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
                match self.settings_active:
                    case 0:
                        self.state = State.ADVANCE_SETTING
                    case 1:
                        self.state = State.ADVANCE_RULE_SETTING
                    case _:
                        self.settings_active = 0
            case State.ASK_QUIT:
                self.running = False
            case State.ADVANCE_SETTING:
                match self.advance_setting_active:
                    case 0:
                        self.state = State.WAVE_SELECT
                    case 1:
                        self.state = State.LEVEL_SELECT
                    case 2:
                        self.state = State.CHANNEL_SELECT
                    case _:
                        self.advance_setting_active = 0
            case State.WAVE_SELECT:
                self.fusion_wavelet = self.wavelet_list[self.wave_active]
                self.state = State.ADVANCE_SETTING
                self.saveSettings()
            case State.LEVEL_SELECT:
                self.fusion_level = self.level_active + 1
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
                self.state = State.RULE_SELECT
            case State.CHANNEL_SELECT:
                if self.channel_sel_active:
                    self.fuse_channel = 3
                else:
                    self.fuse_channel = 1
                self.state = State.ADVANCE_SETTING
                self.saveSettings()
            case State.VIEW_PHOTO:
                if self.did_fusion:
                    self.did_fusion = False
                    self.state = State.CAMERA
                else:
                    self.state = State.ASK_PHOTO
                self.viewPhotoSurface = None
            case State.ASK_PHOTO:
                if self.img1_path is None:
                    self.img1_path = self.capturedIMG_path
                    self.img1_focus = self.capturedIMG_focus
                    self.state = State.CAMERA
                else:
                    self.img2_path = self.capturedIMG_path
                    self.img2_focus = self.capturedIMG_focus
                    self.state = State.ASK_FUSE
            case State.ASK_FUSE:
                self.fuse_photos()
                if self.send_data: self.sendPhoto(self.fused_path)
                self.capturedIMG_path = self.fused_path
                self.state = State.VIEW_PHOTO
                self.did_fusion = True

    def createAssets(self):
        """
        Creates the surfaces which will be used.
        save computation.
        """
        self.small_symbol_font = pygame.font.SysFont("Arial", 20)
        self.big_symbol_font = pygame.font.SysFont("Arial", 50)

        self.screenPlaceHolder = pygame.image.load("assets/img_2.png")
        self.settingsIcon = pygame.image.load("assets/settingsicon2.png")
        self.powerIcon = pygame.image.load("assets/power_button.png")
        self.focusValueSurface = self.small_symbol_font.render(f"F:{self.focusValue}",True, colors.WHITE)

        self.reg_symbol_small = pygame.Surface((20,20))
        self.reg_symbol_small.fill(colors.BLACK)
        temp = self.small_symbol_font.render("R",True, colors.WHITE)
        self.reg_symbol_small.blit(temp, (5,0), temp.get_rect())

        self.fuse_symbol_small = pygame.Surface((20,20))
        self.fuse_symbol_small.fill(colors.BLACK)
        temp = self.small_symbol_font.render("F",True, colors.WHITE)
        self.fuse_symbol_small.blit(temp, (5,0), temp.get_rect())

        self.star_symbol_small = pygame.Surface((20,20))
        self.star_symbol_small.fill(colors.BLACK)
        temp = self.small_symbol_font.render("*",True, colors.WHITE)
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
        self.view_text = self.small_symbol_font.render("VIEW", True, colors.WHITE)
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
        self.screen.blit(self.focusValueSurface, (10, 20), self.focusValueSurface.get_rect())

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
        self.screen.blit(self.rule_text, (10,100), self.registration_text.get_rect())

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
        self.screen.blit(self.fusion_text, (10, 0), self.fusion_text.get_rect())
        wave_txt = self.big_symbol_font.render(self.fusion_wavelet, True, colors.GREEN)
        lvl_txt = self.big_symbol_font.render(str(self.fusion_level), True, colors.GREEN)
        ch_txt = self.big_symbol_font.render(str(self.fuse_channel), True, colors.GREEN)
        self.screen.blit(self.wavelet_text, (10, 50), self.wavelet_text.get_rect())
        self.screen.blit(self.level_text, (10, 100), self.level_text.get_rect())
        self.screen.blit(self.channel_text, (10, 150), self.channel_text.get_rect())
        self.screen.blit(wave_txt, (250, 50), wave_txt.get_rect())
        self.screen.blit(lvl_txt, (250, 100), lvl_txt.get_rect())
        self.screen.blit(ch_txt, (250, 150), ch_txt.get_rect())

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
            temp = self.big_symbol_font.render(wave,True,
                                               colors.GREEN if self.fusion_wavelet == wave else colors.WHITE)
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
            temp = self.big_symbol_font.render(str(i + 1),True,
                                               colors.GREEN if self.fusion_level == i+1 else colors.WHITE)
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

        pygame.draw.rect(self.screen, colors.WHITE, (0, 50 + 50 * self.advance_rule_setting_active, 480, 50), 5)

    def renderChannelSelect(self):
        self.screen.blit(self.rule_text, (10, 0), self.rule_text.get_rect())
        temp = self.big_symbol_font.render("1", True, colors.GREEN)
        self.screen.blit(temp, (10, 50), temp.get_rect())
        temp = self.big_symbol_font.render("3", True, colors.GREEN)
        self.screen.blit(temp, (10, 100), temp.get_rect())

        pygame.draw.rect(self.screen, colors.WHITE, (0, 50 + 50 * self.channel_sel_active, 480, 50), 5)

    def capturePhoto(self):
        """
        Captures Photo from Camera.
        does set state to "ASK_PHOTO"
        """
        print("capturing image")
        if self.testcase:
            if self.img1_path is None:
                self.capturedIMG_path = "photos/test/book1.jpg"
                self.capturedIMG_focus = 0
            else:
                self.capturedIMG_path = "photos/test/book2.jpg"
                self.capturedIMG_focus = 0
        else:
            now = datetime.now()
            formatted = f"photos/captured/Cap{now.year}{now.month:02}{now.day:02}{now.hour:02}{now.minute:02}{now.second:02}.png"
            print(formatted)
            self.capturedIMG_path = formatted
            self.capturedIMG_focus = self.focusValue
            self.cam.capture_file(self.capturedIMG_path)
        print(f"file saved, {self.capturedIMG_path}")
        if self.do_Fusion:
            self.state = State.ASK_PHOTO

    def focusFar(self):
        #command v4l2-ctl -d /dev/v4l-subdev1 -c focus_absolute={value} 0-1023
        if 0 < self.focusValue < 100:
            self.focusValue -= 20
        else:
            self.focusValue -= 50
        if self.focusValue < 0:
            self.focusValue = 0
        os.system(f"v4l2-ctl -d /dev/v4l-subdev1 -c focus_absolute={self.focusValue}")
        # print(f"focus:{self.focusValue}")
        #update focus no on screen
        self.focusValueSurface = self.small_symbol_font.render(f"F:{self.focusValue}", True, colors.WHITE)

    def focusNear(self):
        if 0 < self.focusValue < 100:
            self.focusValue += 20
        else:
            self.focusValue += 50
        if self.focusValue > 1000:
            self.focusValue = 1000
        os.system(f"v4l2-ctl -d /dev/v4l-subdev1 -c focus_absolute={self.focusValue}")
        # print(f"focus:{self.focusValue}")
        #update focus no on screen
        self.focusValueSurface = self.small_symbol_font.render(f"F:{self.focusValue}", True, colors.WHITE)

    def loadSettings(self):
        """
        Load settings from the settings file.
        """
        if os.path.exists(self.settings_file_path):
            with open(self.settings_file_path) as file: data = [line.strip() for line in file.readlines()]
            try:
                self.do_Fusion = True if data[0] == "True" else False
                self.do_registration = True if data[1] == "True" else False
                self.fusion_wavelet = data[2] if data[2] in self.wavelet_list else self.wavelet_list[0]
                self.fusion_level = int(data[3]) if data[3].isdigit() else 2
                self.approx_rule = data[4] if data[4] in self.rule_list else self.rule_list[0]
                self.detail_rule = data[5] if data[5] in self.rule_list else self.rule_list[0]
                self.fuse_channel = int(data[6]) if data[6] in ("1", "3") else 3
            except IndexError:
                raise SettingsError(f"Error loading settings. Try Deleting '{self.settings_file_path}' files")
        else:
            # Default settings
            self.do_Fusion = True
            self.do_registration = True
            self.fusion_wavelet = self.wavelet_list[0]
            self.fusion_level = 2
            self.approx_rule = self.rule_list[0]
            self.detail_rule = self.rule_list[0]
            self.fuse_channel = 3

    def saveSettings(self):
        """
        Save the settings changes to Settings file.
        """
        with open(self.settings_file_path, "wt") as file:
            file.write(str(self.do_Fusion)+"\n")
            file.write(str(self.do_registration)+"\n")
            file.write(self.fusion_wavelet+"\n")
            file.write(str(self.fusion_level)+"\n")
            file.write(self.approx_rule+"\n")
            file.write(self.detail_rule+"\n")
            file.write(str(self.fuse_channel)+"\n")

    def fuse_photos(self):
        """
        Fuse the 'img1_path' and 'img2_path' photos together.
        If registration flag is active. also perform registration before fusion.
        """
        if self.do_registration:
            if self.img1_focus > self.img2_focus:
                self.img2_path = Registration.register(self.img1_path, self.img2_path)
                if self.send_data: self.sendPhoto(self.img2_path)
            else:
                self.img1_path = Registration.register(self.img2_path, self.img1_path)
                if self.send_data: self.sendPhoto(self.img1_path)
        self.fused_path = Fusion.fuse(self.img1_path, self.img2_path, self.fusion_wavelet, self.fusion_level,
                                      self.approx_rule, self.detail_rule, self.fuse_channel)
        self.img1_path = None
        self.img1_focus = 0
        self.img2_path = None
        self.img2_focus = 0

    def parse_args(self):
        parser = ArgumentParser()
        parser.add_argument("-f", "--fullscreen", action="store_true",
                            help="opens the app in Fullscreen mode")
        parser.add_argument("-t", "--test", action="store_true",
                            help="opens the app in test moder.' doesn't use camera but use the images in photos/test'")
        parser.add_argument("-s", "--send", default= None,
                            help="sends the photo to device using SCP")

        self.args = parser.parse_args()

    def sendPhoto(self, photo_path):
        username = "darshan"
        host = "10.90.75.38"
        destination = r"D:/BTech project/recieved_photos/"
        cmd = fr'scp "{photo_path}" {username}@{host}:"{destination}"'

        os.system(cmd)

if __name__ == "__main__":
    App()