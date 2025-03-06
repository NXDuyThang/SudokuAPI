import uvicorn
import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import os

app = FastAPI()

# 🔹 Kiểm tra API hoạt động
@app.get("/")
def home():
    return {"message": "Sudoku OCR API is running!"}

# 🔹 Xử lý ảnh để nhận diện Sudoku
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return img, thresh

# 🔹 Tìm viền của Sudoku
def find_sudoku_contour(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sudoku_contour = max(contours, key=cv2.contourArea)
    return sudoku_contour

# 🔹 Cắt khung Sudoku
def extract_grid(img, contour):
    rect = cv2.boundingRect(contour)
    x, y, w, h = rect
    sudoku_grid = img[y:y+h, x:x+w]
    return sudoku_grid

# 🔹 Chia Sudoku thành 81 ô nhỏ
def split_cells(sudoku_grid):
    cell_size = sudoku_grid.shape[0] // 9
    cells = []
    for i in range(9):
        row = []
        for j in range(9):
            x, y = j * cell_size, i * cell_size
            cell = sudoku_grid[y:y+cell_size, x:x+cell_size]
            row.append(cell)
        cells.append(row)
    return cells

# 🔹 Nhận diện chữ số từ mỗi ô
def recognize_numbers(cells):
    board = []
    config = "--oem 3 --psm 6 outputbase digits"
    for row in cells:
        row_numbers = []
        for cell in row:
            text = pytesseract.image_to_string(cell, config=config).strip()
            row_numbers.append(text if text.isdigit() else "0")
        board.append(row_numbers)
    return board

# 🔹 API nhận ảnh, nhận diện số Sudoku và trả về bảng số
@app.post("/recognize/")
async def recognize_sudoku_api(file: UploadFile = File(...)):
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

        # Xóa file tạm
        os.remove(temp_filename)

        return JSONResponse(content={"sudoku_numbers": recognized_board})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# 🔹 Chạy API
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
