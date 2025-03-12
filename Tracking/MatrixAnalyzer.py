import numpy as np

class MatrixAnalyzer:
    def __init__(self, matrix):
        self.matrix = np.array(matrix)
        self.rows,self.cols=self.matrix.shape

    def dfs(self, row, col, visited):
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        stack = [(row, col)]
        visited[row][col] = True
        area = []

        while stack:
            current_row, current_col = stack.pop()
            area.append((current_row, current_col))

            for dr, dc in directions:
                new_row, new_col = current_row + dr, current_col + dc
                if 0 <= new_row < self.rows and 0 <= new_col < self.cols and not visited[new_row][new_col] and \
                        self.matrix[new_row][new_col] == 1:
                    stack.append((new_row, new_col))
                    visited[new_row][new_col] = True

        return area

    def max_connected_areas(self):
        if not np.any(self.matrix):
            return np.zeros_like(self.matrix)

        rows, cols = self.matrix.shape
        visited = np.zeros((rows, cols), dtype=bool)
        areas = []

        for i in range(rows):
            for j in range(cols):
                if self.matrix[i][j] == 1 and not visited[i][j]:
                    area = self.dfs(i, j, visited)
                    areas.append(area)

        areas.sort(key=len, reverse=True)
        max_areas = areas[:1]
        result_matrix = np.zeros_like(self.matrix)

        # Mark the elements corresponding to the maximum connected areas as 1
        for area in max_areas:
            for row, col in area:
                result_matrix[row, col] = 1

        return result_matrix


        return result_matrix