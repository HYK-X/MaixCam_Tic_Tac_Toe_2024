import random

# 三子棋游戏状态管理类
class TicTacToeGame:
    def __init__(self):
        self.board = [None, None, None, None, None, None, None, None, None]  # 棋盘状态，从0开始计数
        self.human_symbol = "X"  # 人类棋子符号
        self.computer_symbol = "O"  # 电脑棋子符号
        self.current_player = None  # 当前玩家
        self.human_first = True  # 默认人类先下
        self.game_over = False  # 游戏是否结束
        self.winner = None  # 胜利者
        
    def initialize_game(self, human_first=True):
        """初始化游戏，设置谁先下棋
        
        参数:
            human_first (bool): 是否人类先下，True为人类先下，False为电脑先下
            
        返回:
            dict: 包含游戏初始状态的字典
        """
        self.human_first = human_first
        self.board = [None, None, None, None, None, None, None, None, None]
        self.game_over = False
        self.winner = None
        self.current_player = self.human_symbol if human_first else self.computer_symbol
        
        # 如果电脑先手，则让电脑先走一步
        computer_move = None
        if not human_first:
            computer_move = self.get_computer_move()
            self.make_move(computer_move)
        
        return {
            "status": "initialized",
            "current_player": self.current_player,
            "board": self.get_board_representation(),
            "computer_move": computer_move,
            "game_over": self.game_over,
            "winner": self.winner
        }
        
    def make_human_move(self, position):
        """处理人类玩家的落子
        
        参数:
            position (int): 落子位置，0-8
            
        返回:
            dict: 包含游戏状态和电脑响应的字典
        """
        if self.game_over:
            return {
                "status": "error",
                "message": "Game is already over",
                "game_over": self.game_over,
                "winner": self.winner
            }
            
        if self.current_player != self.human_symbol:
            return {
                "status": "error",
                "message": "Not human's turn",
                "current_player": self.current_player
            }
        
        if not (0 <= position <= 8) or self.board[position] is not None:
            return {
                "status": "error",
                "message": "Invalid move",
                "position": position
            }
        
        # 执行人类走棋
        self.make_move(position)
        
        # 检查游戏是否结束
        if self.game_over:
            return {
                "status": "success",
                "human_move": position,
                "board": self.get_board_representation(),
                "game_over": True,
                "winner": self.winner
            }
        
        # 执行电脑走棋
        computer_move = self.get_computer_move()
        self.make_move(computer_move)
        
        return {
            "status": "success",
            "human_move": position,
            "computer_move": computer_move,
            "board": self.get_board_representation(),
            "game_over": self.game_over,
            "winner": self.winner
        }
    
    def update_board_state(self, board_state):
        """根据外部提供的棋盘状态更新游戏
        
        参数:
            board_state (list): 长度为9的数组，表示当前棋盘状态
                                可以是None(空)、'X'(人类)、'O'(电脑)
                                
        返回:
            dict: 包含游戏状态和电脑响应的字典
        """
        if len(board_state) != 9:
            return {
                "status": "error",
                "message": "Invalid board state length"
            }
        
        # 计算棋子数量
        x_count = sum(1 for cell in board_state if cell == 'X')
        o_count = sum(1 for cell in board_state if cell == 'O')
        
        # 验证棋盘状态是否合法
        if self.human_first and not (x_count == o_count or x_count == o_count + 1):
            return {
                "status": "error",
                "message": "Invalid piece count for human first"
            }
        
        if not self.human_first and not (o_count == x_count or o_count == x_count + 1):
            return {
                "status": "error",
                "message": "Invalid piece count for computer first"
            }
        
        # 更新棋盘状态
        self.board = board_state.copy()
        
        # 确定当前应该谁走棋
        if self.human_first:
            self.current_player = self.human_symbol if x_count == o_count else self.computer_symbol
        else:
            self.current_player = self.human_symbol if o_count > x_count else self.computer_symbol
        
        # 检查游戏是否已经结束
        self.check_winner()
        
        if self.game_over:
            return {
                "status": "success",
                "board": self.get_board_representation(),
                "game_over": True,
                "winner": self.winner
            }
        
        # 如果轮到电脑走棋，则计算下一步
        computer_move = None
        if self.current_player == self.computer_symbol:
            computer_move = self.get_computer_move()
            self.make_move(computer_move)
        
        return {
            "status": "success",
            "computer_move": computer_move,
            "board": self.get_board_representation(),
            "game_over": self.game_over,
            "winner": self.winner
        }
    
    def get_board_representation(self):
        """获取棋盘表示，便于可视化
        
        返回:
            list: 棋盘状态列表
        """
        return self.board.copy()
    
    def make_move(self, position):
        """在指定位置落子
        
        参数:
            position (int): 落子位置，0-8
            
        返回:
            bool: 是否成功落子
        """
        if position < 0 or position > 8 or self.board[position] is not None or self.game_over:
            return False
        
        self.board[position] = self.current_player
        
        # 检查是否有胜利者
        if not self.check_winner():
            # 切换玩家
            self.current_player = self.computer_symbol if self.current_player == self.human_symbol else self.human_symbol
            
        return True
    
    def check_winner(self):
        """检查是否有胜利者
        
        返回:
            bool: 游戏是否结束
        """
        # 定义所有可能的获胜组合
        win_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # 横向
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # 纵向
            [0, 4, 8], [2, 4, 6]  # 对角线
        ]
        
        for combo in win_combinations:
            if (self.board[combo[0]] is not None and
                self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]]):
                self.game_over = True
                self.winner = self.board[combo[0]]
                return True
        
        # 检查是否平局
        if None not in self.board:
            self.game_over = True
            self.winner = "Draw"
            return True
            
        return False
    
    def get_computer_move(self):
        """使用Minimax算法获取电脑的下一步落子位置
        
        返回:
            int: 电脑选择的落子位置
        """
        # 如果棋盘为空（第一步），选择中心位置或随机角落
        if all(cell is None for cell in self.board):
            return 4  # 中心位置
        
        # 使用Minimax算法寻找最佳位置
        best_score = float('-inf')
        best_move = -1
        
        for i in range(9):
            if self.board[i] is None:
                # 尝试在位置i落子
                self.board[i] = self.computer_symbol
                # 计算这步棋的分数
                score = self.minimax(0, False)
                # 撤销这步棋
                self.board[i] = None
                
                # 更新最佳分数和最佳移动
                if score > best_score:
                    best_score = score
                    best_move = i
        
        return best_move
    
    def minimax(self, depth, is_maximizing):
        """Minimax算法实现
        
        参数:
            depth (int): 当前递归深度
            is_maximizing (bool): 是否是最大化玩家（电脑）的回合
            
        返回:
            int: 当前棋盘状态的评分
        """
        # 检查是否有赢家
        if self._check_win_condition(self.board, self.computer_symbol):
            return 10 - depth  # 电脑赢，分数越高越好，深度越浅越好
        if self._check_win_condition(self.board, self.human_symbol):
            return depth - 10  # 人类赢，分数越低越好
        if None not in self.board:
            return 0  # 平局
        
        if is_maximizing:  # 电脑回合，寻找最高分数
            best_score = float('-inf')
            for i in range(9):
                if self.board[i] is None:
                    self.board[i] = self.computer_symbol
                    score = self.minimax(depth + 1, False)
                    self.board[i] = None
                    best_score = max(score, best_score)
            return best_score
        else:  # 人类回合，寻找最低分数
            best_score = float('inf')
            for i in range(9):
                if self.board[i] is None:
                    self.board[i] = self.human_symbol
                    score = self.minimax(depth + 1, True)
                    self.board[i] = None
                    best_score = min(score, best_score)
            return best_score
    
    def _check_win_condition(self, board, symbol):
        """检查指定符号是否获胜
        
        参数:
            board (list): 棋盘状态
            symbol (str): 要检查的棋子符号
            
        返回:
            bool: 是否获胜
        """
        win_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # 横向
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # 纵向
            [0, 4, 8], [2, 4, 6]  # 对角线
        ]
        
        for combo in win_combinations:
            if board[combo[0]] == board[combo[1]] == board[combo[2]] == symbol:
                return True
        return False


# 使用示例
if __name__ == "__main__":
    # 创建游戏实例
    game = TicTacToeGame()
    
    # 示例1: 初始化游戏，人类先手
    print("示例1: 人类先手")
    result = game.initialize_game(human_first=True)
    print(result)
    
    # 示例2: 人类走棋
    print("\n示例2: 人类走棋在位置0（左上角）")
    result = game.make_human_move(0)
    print(result)
    
    # 示例3: 外部更新棋盘状态（模拟视觉识别）
    print("\n示例3: 外部更新棋盘状态")
    new_board = [None, None, "X", None, "O", None, None, None, "X"]
    result = game.update_board_state(new_board)
    print(result)