class BoardChangeDetector:
    """
    井字棋棋局变动检测类 - 简化版
    维护当前棋盘状态，并能检测新棋盘状态相对于当前状态的变化
    返回简化的结果格式
    """
    
    def __init__(self, initial_board=None):
        """
        初始化检测器，设置初始棋盘状态
        
        Args:
            initial_board (list, optional): 初始棋盘状态，为9个元素的列表。
                                           如果未提供，则默认为空棋盘。
        """
        if initial_board is None:
            self.current_board = [None] * 9
        else:
            if len(initial_board) != 9:
                raise ValueError("棋盘必须包含9个元素")
            self.current_board = initial_board.copy()
    
    def get_current_board(self):
        """获取当前保存的棋盘状态"""
        return self.current_board.copy()
    
    def _count_pieces(self, board):
        """计算棋盘上X和O的数量"""
        x_count = board.count("X")
        o_count = board.count("O")
        return x_count, o_count
    
    def _get_positions(self, board, piece):
        """获取特定棋子在棋盘上的位置"""
        return [i for i, val in enumerate(board) if val == piece]
    
    def detect_change(self, new_board):
        """
        检测新棋盘状态相对于当前保存的棋盘状态的变化，返回简化的结果
        
        Args:
            new_board (list): 新棋盘状态，表示为9个元素的列表
            
        Returns:
            dict: 包含简化变化检测结果的字典
        """
        if len(new_board) != 9:
            raise ValueError("棋盘必须包含9个元素")
        
        old_board = self.current_board
        
        # 获取X和O的数量
        old_x_count, old_o_count = self._count_pieces(old_board)
        new_x_count, new_o_count = self._count_pieces(new_board)
        
        # 获取X和O的位置
        old_x_positions = self._get_positions(old_board, "X")
        new_x_positions = self._get_positions(new_board, "X")
        old_o_positions = self._get_positions(old_board, "O")
        new_o_positions = self._get_positions(new_board, "O")
        
        # 检查变动
        x_added = [pos for pos in new_x_positions if pos not in old_x_positions]
        x_removed = [pos for pos in old_x_positions if pos not in new_x_positions]
        o_added = [pos for pos in new_o_positions if pos not in old_o_positions]
        o_removed = [pos for pos in old_o_positions if pos not in new_o_positions]
        
        result = {"status": "不变"}
        
        # 检查是否有新增棋子
        if x_added and not x_removed:
            result["status"] = "新增"
            result["piece"] = "X"
            result["position"] = x_added[0]  # 通常只会新增一个
            
        elif o_added and not o_removed:
            result["status"] = "新增"
            result["piece"] = "O"
            result["position"] = o_added[0]
            
        # 检查是否有移动棋子（有移除也有新增）
        elif (x_added and x_removed) or (o_added and o_removed):
            result["status"] = "移动"
            if x_added and x_removed:
                result["piece"] = "X"
                result["from_pos"] = x_removed[0]
                result["to_pos"] = x_added[0]
            else:
                result["piece"] = "O"
                result["from_pos"] = o_removed[0]
                result["to_pos"] = o_added[0]
        
        return result
    
    def update_board(self, new_board):
        """
        更新当前保存的棋盘状态，并返回变化情况
        
        Args:
            new_board (list): 新棋盘状态，表示为9个元素的列表
            
        Returns:
            dict: 包含简化变化检测结果的字典
        """
        result = self.detect_change(new_board)
        self.current_board = new_board.copy()
        return result


# 使用示例
if __name__ == "__main__":
    # 初始化检测器，设置初始棋盘
    initial_board = [None, None, "X", None, "O", None, None, None, "X"]
    detector = BoardChangeDetector(initial_board)
    
    # 测试1：新增一个X
    new_board1 = [None, "X", "X", None, "O", None, None, None, "X"]
    result1 = detector.update_board(new_board1)
    print("新增X的变化:", result1)
    # 预期输出: {'status': '新增', 'piece': 'X', 'position': 1}
    
    # 测试2：移动O
    new_board2 = [None, "X", "X", None, None, "O", None, None, "X"]
    result2 = detector.update_board(new_board2)
    print("移动O的变化:", result2)
    # 预期输出: {'status': '移动', 'piece': 'O', 'from_pos': 4, 'to_pos': 5}
    
    # 测试3：不变
    new_board3 = [None, "X", "X", None, None, "O", None, None, "X"]
    result3 = detector.update_board(new_board3)
    print("不变的情况:", result3)
    # 预期输出: {'status': '不变'}