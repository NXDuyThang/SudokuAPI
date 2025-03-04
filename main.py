import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import os

app = FastAPI()

# Xử lý ảnh: Tiền xử lý để nhận diện Sudoku
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return img, thresh

# Tìm khung Sudoku lớn nhất trong ảnh
def find_sudoku_contour(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sudoku_contour = max(contours, key=cv2.contourArea)
    return sudoku_contour

# Cắt khung Sudoku ra khỏi ảnh
def extract_grid(img, contour):
    rect = cv2.boundingRect(contour)
    x, y, w, h = rect
    sudoku_grid = img[y:y+h, x:x+w]
    return sudoku_grid

# Chia Sudoku thành 81 ô nhỏ (9x9)
def split_cells(sudoku_grid):
    cell_size = sudoku_grid.shape[0] // 9
    cells = []
    for i in range(9):
        for j in range(9):
            x, y = j * cell_size, i * cell_size
            cell = sudoku_grid[y:y+cell_size, x:x+cell_size]
            cells.append(cell)
    return cells

# Nhận diện chữ số bằng Tesseract OCR
def recognize_numbers(cells):
    board = []
    config = "--oem 3 --psm 6 outputbase digits"
    for cell in cells:
        text = pytesseract.image_to_string(cell, config=config).strip()
        board.append(text if text.isdigit() else "0")
    return np.array(board).reshape(9, 9).tolist()

# Route kiểm tra API
@app.get("/")
def home():
    return {"message": "Sudoku OCR API is running!"}

# Route nhận ảnh Sudoku & trả về kết quả
@app.post("/ocr/")
async def ocr_sudoku(file: UploadFile = File(...)):
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

        # Xóa file tạm sau khi xử lý
        os.remove(temp_filename)

        return JSONResponse(content={"sudoku_numbers": recognized_board})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
