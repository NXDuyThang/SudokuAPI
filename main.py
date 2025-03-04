import uvicorn
import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import os

app = FastAPI()

# 🔹 Route kiểm tra API
@app.get("/")
def home():
    return {"message": "Sudoku OCR & Solver API is running!"}

# 🔹 Xử lý ảnh: Tiền xử lý để nhận diện Sudoku
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return img, thresh

# 🔹 Tìm khung Sudoku lớn nhất
def find_sudoku_contour(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sudoku_contour = max(contours, key=cv2.contourArea)
    return sudoku_contour

# 🔹 Cắt khung Sudoku ra khỏi ảnh
def extract_grid(img, contour):
    rect = cv2.boundingRect(contour)
    x, y, w, h = rect
    sudoku_grid = img[y:y+h, x:x+w]
    return sudoku_grid

# 🔹 Chia Sudoku thành 81 ô nhỏ (9x9)
def split_cells(sudoku_grid):
    cell_size = sudoku_grid.shape[0] // 9
    cells = []
    for i in range(9):
        for j in range(9):
            x, y = j * cell_size, i * cell_size
            cell = sudoku_grid[y:y+cell_size, x:x+cell_size]
            cells.append(cell)
    return cells

# 🔹 Nhận diện chữ số bằng Tesseract OCR
def recognize_numbers(cells):
    board = []
    config = "--oem 3 --psm 6 outputbase digits"
    for cell in cells:
        text = pytesseract.image_to_string(cell, config=config).strip()
        board.append(text if text.isdigit() else "0")
    return np.array(board).reshape(9, 9).tolist()

# 🔹 Thuật toán Backtracking giải Sudoku
def is_valid(board, row, col, num):
    """ Kiểm tra xem có thể đặt số vào ô không """
    num = str(num)
    
    # Kiểm tra hàng, cột
    for i in range(9):
        if board[row][i] == num or board[i][col] == num:
            return False

    # Kiểm tra trong khối 3x3
    start_row, start_col = (row // 3) * 3, (col // 3) * 3
    for i in range(3):
        for j in range(3):
            if board[start_row + i][start_col + j] == num:
                return False
    return True

def solve_sudoku(board):
    """ Hàm Backtracking để giải Sudoku """
    for row in range(9):
        for col in range(9):
            if board[row][col] == "0":
                for num in range(1, 10):
                    if is_valid(board, row, col, num):
                        board[row][col] = str(num)
                        if solve_sudoku(board):
                            return True
                        board[row][col] = "0"  # Quay lui nếu không hợp lệ
                return False  # Không có số nào hợp lệ
    return True

# 🔹 Route nhận ảnh & giải Sudoku
@app.post("/solve/")
async def solve_sudoku_api(file: UploadFile = File(...)):
    temp_filename = f"temp_{file.filename}"
    
    # Lưu file ảnh tạm thời
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Xử lý ảnh
        img, thresh = preprocess_image(temp_filename)
        sudoku_contour = find_sudoku_contour(thresh)
        sudoku_grid = extract_grid(img, sudoku_contour)
        cells = split_cells(sudoku_grid)
        recognized_board = recognize_numbers(cells)

        # Giải Sudoku bằng Backtracking
        solved_board = [row[:] for row in recognized_board]  # Copy bảng để giải
        if solve_sudoku(solved_board):
            result = {"sudoku_numbers": recognized_board, "solved_sudoku": solved_board}
        else:
            result = {"error": "Không thể giải Sudoku!"}

        # Xóa file tạm
        os.remove(temp_filename)

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# 🔹 Chạy API trên Render
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
