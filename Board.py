import torch


SHEEP = 2
WOLF = 1
EMPTY = 0

ONGOING_GAME = 0
WOLF_WON = 1
SHEEP_WON = 2

WOLF_UR = 0
WOLF_UL = 1
WOLF_DR = 2
WOLF_DL = 3

class Board:
    def __init__(self) -> None:
        # 8x8 board
        self.matrix = torch.zeros(8, 8, dtype=torch.float32)
        self.wolf_pos = None
        self.sheep_pos = []

    # Print the board to terminal
    def print(self) -> None:
        # clear entire screen
        print("\033c", end="")

        for row in self.matrix:
            for slot in row:
                val = slot.item()
                if val == WOLF:
                    print('W', end=' ')
                elif val == EMPTY:
                    print('.', end=' ')
                else:
                    print('S', end=' ')
            print()

    def reset(self) -> None:
        self.matrix = torch.zeros(8, 8, dtype=torch.float32)
        self.sheep_pos = []
        self.move_limit = 50

        # Sheep positions
        sheep_counter = 2
        for i in range(1, 8, 2):
            self.sheep_pos.append((0, i))
            self.matrix[0][i] = sheep_counter
            sheep_counter += 1

        # Wolf position
        middle_index = self.matrix.shape[1] // 2
        self.matrix[-1, middle_index] = WOLF
        self.wolf_pos = (7, middle_index)

    # convert 8x8 tensor to 64x1 tensor
    def to_state(self):
      # make a copy of the matrix and flatten it
        return self.matrix.flatten()
    # check if the path is valid from the origin
    def check_valid_path(self, orig: tuple, d_y: int, d_x: int) -> bool:
      new_y = orig[0] + d_y
      new_x = orig[1] + d_x

      if 0 <= new_x <= 7 and 0 <= new_y <= 7:
        return self.matrix[new_y, new_x].item() == EMPTY
      else:
        return False

    # check if the wolf is stuck (sheep win condition)
    def check_wolf_stuck(self) -> bool:
      for d_y, d_x in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        # Check if the diagonal path is blocked using the existing method
        if self.check_valid_path(self.wolf_pos, d_y, d_x):
          # If any diagonal path is not blocked, return False (wolf can move)
          return False
      # wolf is stuck
      return True

    # check if the wolf passed all the sheep (wolf win condition)
    def check_sheep_defense(self) -> bool:
        for sheep in self.sheep_pos:
            if sheep[0] <= self.wolf_pos[0]:
                return False
        return True

    def move(self, orig: tuple, dir: tuple) -> tuple:
      d_y, d_x = dir
      y, x = orig
      val = self.matrix[y, x].item()
      if self.check_valid_path(orig, d_y, d_x):
        self.matrix[y, x] = EMPTY
        self.matrix[y + d_y, x + d_x] = val
        return (y + d_y, x + d_x), False
      return orig, True

    def move_wolf(self, wolf_action: int) -> bool:
        dirs = [(-1, 1), (-1, -1), (1, 1), (1, -1)]
        self.wolf_pos, collide = self.move(self.wolf_pos, dirs[wolf_action])
        return collide

    def move_sheep(self, sheep_action: int) -> bool:
        sheep_id = sheep_action // 2
        dir_id = sheep_action % 2
        dirs = [(1, 1), (1, -1)]

        self.sheep_pos[sheep_id],collide = self.move(self.sheep_pos[sheep_id], dirs[dir_id])
        return collide


    def get_reward(self) -> int:
      if self.check_wolf_stuck():
        return SHEEP_WON
      elif self.check_sheep_defense():
        return WOLF_WON
      else:
        return ONGOING_GAME
