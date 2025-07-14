from maix import display, image, touchscreen, time, camera

class UIElement:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.visible = True
        self.page = None
        
    def draw(self, img):
        pass
        
    def contains(self, x, y):
        return (self.x <= x <= self.x + self.w) and (self.y <= y <= self.y + self.h)

class Button(UIElement):
    def __init__(self, x, y, w, h, label="", callback=None):
        super().__init__(x, y, w, h)
        self.label = label
        self.callback = callback
        self.color = image.COLOR_WHITE
        self.bg_color = None
        self.pressed = False
        self.font_scale = 1.5
        self.char_width = 10
        self.line_height = 20

    def _estimate_text_size(self):
        text_width = len(self.label) * self.char_width * self.font_scale
        text_height = self.line_height * self.font_scale
        return int(text_width), int(text_height)

    def _auto_adjust_font(self):
        max_width = self.w * 0.8
        while self.font_scale > 0.8:
            text_w, _ = self._estimate_text_size()
            if text_w <= max_width:
                break
            self.font_scale -= 0.1
        self.font_scale = max(0.8, round(self.font_scale, 1))

    def draw(self, img):
        if not self.visible:
            return
        self._auto_adjust_font()
        border_color = image.COLOR_RED if self.pressed else self.color
        img.draw_rect(self.x, self.y, self.w, self.h, border_color, 2)
        text_w, text_h = self._estimate_text_size()
        text_x = self.x + (self.w - text_w) // 2
        text_y = self.y + (self.h - text_h) // 2 + int(text_h * 0.3)
        img.draw_string(text_x+1, text_y+1, self.label,
                      color=image.COLOR_BLACK, scale=self.font_scale)
        text_color = image.COLOR_GRAY if self.pressed else self.color
        img.draw_string(text_x, text_y, self.label,
                      color=text_color, scale=self.font_scale)

class MenuPage:
    def __init__(self, title="", id=None):
        self.elements = []
        self.title = title
        self.id = id or title  # 使用id标识页面，默认为title
        self.parent = None
        self.children = {}  # 使用字典而不是列表，键为页面ID
        self.visible = False
        
    def add_element(self, element):
        element.page = self
        self.elements.append(element)
    
    def add_child(self, child_page):
        """添加子页面"""
        child_page.parent = self
        self.children[child_page.id] = child_page
        return child_page
    
    def get_child(self, id):
        """通过ID获取子页面"""
        return self.children.get(id)
        
    def show(self):
        """显示当前页面"""
        self.visible = True
        for element in self.elements:
            element.visible = True
            
    def hide(self):
        """隐藏当前页面"""
        self.visible = False
        for element in self.elements:
            element.visible = False
    
    def get_path(self):
        """获取从根到当前页面的路径"""
        if not self.parent:
            return [self.title]
        return self.parent.get_path() + [self.title]

class EZGUI:
    def __init__(self):
        self.ts = touchscreen.TouchScreen()
        self.disp = display.Display()
        self.root_page = MenuPage("Root", "root")
        self.current_page = None
        self.last_pressed = False
        self.title_height = 30
        self.bottom_margin = 60
        self.active_button = None
        self.page_cache = {}  # 页面缓存，通过路径查找页面
        
        # 初始化主页面
        self.main_page = self.create_page("Main", "main", self.root_page)
        self.show_page(self.main_page)
        
    def create_page(self, title, id=None, parent=None):
        """创建新页面并添加到父页面"""
        page_id = id or title
        
        # 检查缓存中是否已存在此页面
        if page_id in self.page_cache:
            return self.page_cache[page_id]
            
        page = MenuPage(title, page_id)
        if parent:
            parent.add_child(page)
        
        # 添加到缓存
        self.page_cache[page_id] = page
        return page
    
    def get_page(self, page_id):
        """通过ID获取页面"""
        return self.page_cache.get(page_id)
    
    def show_page(self, page):
        """显示指定页面，隐藏当前页面"""
        if self.current_page:
            self.current_page.hide()
        self.current_page = page
        page.show()
        
        # 打印当前路径
        print("Current path:", " > ".join(page.get_path()))
    
    def navigate_to_path(self, path):
        """通过路径导航到页面"""
        if not path:
            return False
            
        current = self.root_page
        for page_id in path[1:]:  # 跳过根页面
            next_page = current.get_child(page_id)
            if not next_page:
                return False
            current = next_page
            
        self.show_page(current)
        return True
    
    def back(self):
        """返回父页面"""
        if self.current_page and self.current_page.parent:
            self.show_page(self.current_page.parent)
    
    def add_buttons(self, positions, labels, callbacks, 
                   col_spacing=10, row_spacing=10, 
                   bottom_margin=None):
        """添加多个按钮到当前页面"""
        if not self.current_page:
            return
            
        used_bottom_margin = self.bottom_margin if bottom_margin is None else bottom_margin
        
        cols, rows = positions
        available_h = self.disp.height() - self.title_height - used_bottom_margin - (rows-1)*row_spacing
        btn_h = available_h // rows
        available_w = self.disp.width() - (cols-1)*col_spacing
        btn_w = available_w // cols
        
        for row in range(rows):
            for col in range(cols):
                idx = row * cols + col
                if idx >= len(labels):
                    return
                x = col * (btn_w + col_spacing)
                y = self.title_height + row * (btn_h + row_spacing)
                btn = Button(x, y, btn_w, btn_h, labels[idx], callbacks[idx])
                self.current_page.add_element(btn)

    def add_back_button(self, text="Back", size=(80, 40)):
        """添加返回按钮"""
        if not self.current_page:
            return
            
        btn = Button(10, 
                    self.disp.height() - self.bottom_margin + (self.bottom_margin - size[1])//2,
                    size[0], size[1], 
                    text, self.back)
        self.current_page.add_element(btn)

    def add_exit_button(self, text="Exit", size=(80, 40)):
        """添加退出按钮"""
        if not self.current_page:
            return
            
        btn = Button(self.disp.width() - size[0] - 10,
                    self.disp.height() - self.bottom_margin + (self.bottom_margin - size[1])//2,
                    size[0], size[1], 
                    text, lambda: exit())
        self.current_page.add_element(btn)

    def update(self, bg_img=None):
        """更新UI状态和绘制"""
        if not self.current_page:
            return
            
        x, y, pressed = self.ts.read()
        
        # 更新按钮的按下状态
        for element in self.current_page.elements:
            if isinstance(element, Button) and element.visible:
                if element.contains(x, y):
                    if pressed and not self.last_pressed:  # 按下时
                        element.pressed = True
                        self.active_button = element
                    elif not pressed and self.last_pressed:  # 释放时
                        if element.pressed and element.callback:
                            element.callback()
                        element.pressed = False
                        self.active_button = None
                else:
                    element.pressed = False  # 确保其他按钮恢复默认状态

        self.last_pressed = pressed

        # 背景图处理逻辑
        if bg_img:
            if bg_img.width() != self.disp.width() or bg_img.height() != self.disp.height():
                scale_x = self.disp.width() / bg_img.width()
                scale_y = self.disp.height() / bg_img.height()
                scale = min(scale_x, scale_y)
                
                new_width = int(bg_img.width() * scale)
                new_height = int(bg_img.height() * scale)
                bg_img = bg_img.resize(new_width, new_height)

                img = image.Image(self.disp.width(), self.disp.height())
                offset_x = (self.disp.width() - new_width) // 2
                offset_y = (self.disp.height() - new_height) // 2
                img.draw_image(offset_x, offset_y, bg_img)
        else:
            img = image.Image(self.disp.width(), self.disp.height())

        # UI绘制逻辑
        if self.current_page.title:
            img.draw_string(10, 5, self.current_page.title, 
                           color=image.COLOR_WHITE, scale=2)
        
        for element in self.current_page.elements:
            if element.visible:
                element.draw(img)
                
        self.disp.show(img)
        time.sleep_ms(50)
