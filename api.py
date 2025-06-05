from flask import Flask, request, jsonify
import cv2
import torch
import function.utils_rotate as utils_rotate
import function.helper as helper
import os
import base64
import numpy as np
import time
from werkzeug.utils import secure_filename
from flask_cors import CORS
from plate_memory import PlateMemoryManager

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Cấu hình thư mục lưu file tạm
UPLOAD_FOLDER = 'temp'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# API chỉ xử lý nhận diện biển số, không xử lý database
# Tối ưu hóa cho performance và memory

import threading
import queue
import gc

# Biến global cho models
yolo_LP_detect = None
yolo_license_plate = None

# Khởi tạo PlateMemoryManager với ngưỡng similarity thấp hơn cho API
plate_memory = PlateMemoryManager()
plate_memory.similarity_threshold = 0.2  # Lowered for better detection  # Giảm từ 0.6 xuống 0.5 để tăng khả năng nhận dạng cho camera real-time

# Queue để xử lý request
request_queue = queue.Queue(maxsize=10)  # Giới hạn 10 request đồng thời
processing_lock = threading.Lock()

# Load models với model tối ưu hóa
def load_models():
    global yolo_LP_detect, yolo_license_plate
    print("Loading AI models...")

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector.pt',
                                   force_reload=True, source='local', trust_repo=True)
    yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr.pt',
                                       force_reload=True, source='local', trust_repo=True)

    # Tối ưu hóa models
    yolo_LP_detect.conf = 0.25  # Confidence threshold
    yolo_license_plate.conf = 0.60

    # Warm up models
    print("Warming up models...")
    dummy_img = torch.zeros((1, 3, 640, 640))
    with torch.no_grad():
        _ = yolo_LP_detect(dummy_img)

    print("✅ Models loaded and optimized successfully!")

# Load models khi khởi động
load_models()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def base64_to_image(base64_string):
    """Chuyển đổi base64 string thành OpenCV image"""
    try:
        # Loại bỏ header data:image/jpeg;base64, nếu có
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]

        # Decode base64
        img_data = base64.b64decode(base64_string)

        # Chuyển đổi thành numpy array
        nparr = np.frombuffer(img_data, np.uint8)

        # Decode thành OpenCV image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        return img
    except Exception as e:
        print(f"Lỗi chuyển đổi base64: {e}")
        return None

def normalize_plate_text(plate_text):
    """Chuẩn hóa format biển số để so sánh tốt hơn"""
    if not plate_text:
        return ""

    # Loại bỏ khoảng trắng và chuyển về uppercase
    normalized = plate_text.strip().upper()

    # Thêm dấu gạch ngang nếu chưa có (format chuẩn: XX-YYYY)
    if '-' not in normalized and len(normalized) >= 6:
        # Tìm vị trí số đầu tiên
        for i, char in enumerate(normalized):
            if char.isdigit():
                if i > 0:  # Có chữ cái trước số
                    normalized = normalized[:i] + '-' + normalized[i:]
                break

    return normalized

def check_plate_memory(crop_img):
    """Kiểm tra plate memory để tìm vùng biển số tương tự đã được gán nhãn"""
    try:
        similar_match = plate_memory.find_similar_plate(crop_img)
        if similar_match:
            plate_id, plate_text, similarity = similar_match
            # Chuẩn hóa plate text
            normalized_text = normalize_plate_text(plate_text)
            print(f"🎯 Plate Memory: Tìm thấy {normalized_text} (similarity: {similarity:.2f})")
            return {
                'found': True,
                'plate_text': normalized_text,
                'similarity': similarity,
                'plate_id': plate_id
            }
        return {'found': False}
    except Exception as e:
        print(f"Lỗi khi kiểm tra plate memory: {e}")
        return {'found': False}

def save_unrecognized_plate(crop_img, source_info="API"):
    """Lưu vùng biển số không nhận dạng được vào plate memory"""
    try:
        plate_id = plate_memory.save_unrecognized_plate(crop_img, source_info)
        return {
            'saved': True,
            'plate_id': plate_id
        }
    except Exception as e:
        print(f"Lỗi khi lưu vào plate memory: {e}")
        return {'saved': False}

def detect_license_plate_from_image(img):
    """Nhận diện biển số từ OpenCV image - tối ưu hóa memory và performance"""
    if img is None:
        return {
            'success': False,
            'message': 'Ảnh không hợp lệ',
            'plates': [],
            'confidence': 0,
            'processing_time': 0
        }

    start_time = time.time()

    try:
        with processing_lock:  # Đảm bảo chỉ xử lý 1 request tại 1 thời điểm
            # Cải thiện ảnh chất lượng thấp
            height, width = img.shape[:2]

            # Nếu ảnh quá nhỏ, upscale lên
            if width < 800 or height < 600:
                scale_factor = max(800/width, 600/height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                print(f"📈 Upscaled image: {width}x{height} → {new_width}x{new_height}")

            # Cải thiện contrast và brightness cho ảnh tối
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            img = cv2.merge([l, a, b])
            img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

            # Phát hiện biển số với torch.no_grad() để tiết kiệm memory
            with torch.no_grad():
                plates = yolo_LP_detect(img, size=640)
                list_plates = plates.pandas().xyxy[0].values.tolist()

            list_read_plates = set()
            confidence_scores = []
            plate_memory_results = []  # Lưu kết quả từ plate memory

            if len(list_plates) == 0:
                print("⚠️  Không phát hiện biển số, thử các phương pháp fallback...")

                # Fallback 1: Thử đọc toàn bộ ảnh
                with torch.no_grad():
                    lp = helper.read_plate(yolo_license_plate, img)
                if lp != "unknown":
                    clean_plate = lp.strip().upper()
                    if len(clean_plate) >= 6:
                        list_read_plates.add(clean_plate)
                        confidence_scores.append(50)
                        print(f"🔄 Fallback OCR toàn ảnh: {clean_plate}")

                # Fallback 2: Thử với YOLO threshold thấp hơn
                if len(list_read_plates) == 0:
                    with torch.no_grad():
                        plates_low = yolo_LP_detect(img, size=640)
                        # Lọc với confidence thấp hơn (0.3 thay vì 0.5)
                        list_plates_low = plates_low.pandas().xyxy[0]
                        list_plates_low = list_plates_low[list_plates_low['confidence'] > 0.3].values.tolist()

                    if len(list_plates_low) > 0:
                        print(f"🔄 Fallback với threshold thấp: tìm thấy {len(list_plates_low)} vùng")
                        list_plates = list_plates_low  # Sử dụng kết quả threshold thấp
            else:
                for plate in list_plates:
                    x = int(plate[0])
                    y = int(plate[1])
                    w = int(plate[2] - plate[0])
                    h = int(plate[3] - plate[1])

                    # Kiểm tra kích thước crop hợp lệ
                    if w > 10 and h > 10:
                        crop_img = img[y:y+h, x:x+w]
                        plate_confidence = float(plate[4]) * 100  # Confidence từ YOLO detection

                        # ✨ ƯU TIÊN PLATE MEMORY TRƯỚC - Kiểm tra plate memory đầu tiên
                        found = False
                        memory_result = check_plate_memory(crop_img)

                        if memory_result['found']:
                            # 🎯 Tìm thấy trong plate memory - ưu tiên cao nhất
                            plate_text = memory_result['plate_text']
                            similarity = memory_result['similarity']
                            list_read_plates.add(plate_text)
                            # Confidence cao cho plate memory (similarity * 100 + bonus)
                            memory_confidence = min(similarity * 100 + 10, 100)  # Bonus 10 điểm, max 100
                            confidence_scores.append(memory_confidence)
                            plate_memory_results.append({
                                'plate_text': plate_text,
                                'similarity': similarity,
                                'plate_id': memory_result['plate_id'][:8],
                                'source': 'plate_memory',
                                'priority': 'high'  # Đánh dấu ưu tiên cao
                            })
                            found = True
                            print(f"🎯 API: Ưu tiên từ plate memory: {plate_text} (similarity: {similarity:.2f}, confidence: {memory_confidence:.1f})")

                        # Nếu không tìm thấy trong memory, mới thử OCR
                        if not found:
                            for cc in range(0, 2):
                                if found:
                                    break
                                for ct in range(0, 2):
                                    with torch.no_grad():
                                        lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                                    if lp != "unknown":
                                        clean_plate = lp.strip().upper()
                                        # Kiểm tra định dạng biển số Việt Nam cơ bản
                                        if len(clean_plate) >= 6:
                                            list_read_plates.add(clean_plate)
                                            confidence_scores.append(plate_confidence)
                                            found = True
                                            print(f"🤖 API: OCR nhận dạng: {clean_plate} (confidence: {plate_confidence:.1f})")
                                            break

            processing_time = time.time() - start_time

            # ✨ Sắp xếp kết quả ưu tiên plate memory trước
            result_plates = list(list_read_plates)

            # Sắp xếp theo ưu tiên: plate memory trước, OCR sau
            if plate_memory_results:
                # Tách biển số từ memory và OCR
                memory_plates = [r['plate_text'] for r in plate_memory_results]
                ocr_plates = [p for p in result_plates if p not in memory_plates]

                # Sắp xếp lại: memory trước, OCR sau
                result_plates = memory_plates + ocr_plates

                # Tính confidence ưu tiên memory
                memory_confidences = [min(r['similarity'] * 100 + 10, 100) for r in plate_memory_results]
                ocr_confidences = [c for i, c in enumerate(confidence_scores) if i >= len(memory_confidences)]
                confidence_scores = memory_confidences + ocr_confidences

            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

            # Dọn dẹp memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Tạo thông báo chi tiết với ưu tiên
            message_parts = []
            if result_plates:
                memory_count = len(plate_memory_results)
                ocr_count = len(result_plates) - memory_count

                # Ưu tiên hiển thị memory trước
                if memory_count > 0:
                    message_parts.append(f"🎯 Memory: {memory_count} biển số (ưu tiên)")
                if ocr_count > 0:
                    message_parts.append(f"🤖 OCR: {ocr_count} biển số")

                if memory_count > 0:
                    message = f"✅ Phát hiện {len(result_plates)} biển số - Ưu tiên từ Plate Memory ({', '.join(message_parts)})"
                else:
                    message = f"Phát hiện {len(result_plates)} biển số ({', '.join(message_parts)})"
            else:
                message = 'Không phát hiện biển số'

            return {
                'success': True,
                'plates': result_plates,
                'confidence': round(avg_confidence, 2),
                'processing_time': round(processing_time, 3),
                'detected_boxes': len(list_plates),
                'message': message,
                'plate_memory_results': plate_memory_results,  # Thông tin từ plate memory
                'detection_methods': {
                    'ocr_count': len(result_plates) - len(plate_memory_results),
                    'memory_count': len(plate_memory_results),
                    'total_count': len(result_plates)
                }
            }

    except Exception as e:
        processing_time = time.time() - start_time
        # Dọn dẹp memory khi có lỗi
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return {
            'success': False,
            'message': f'Lỗi xử lý: {str(e)}',
            'plates': [],
            'confidence': 0,
            'processing_time': round(processing_time, 3),
            'detected_boxes': 0
        }

def detect_license_plate(image_path):
    """Nhận diện biển số từ file path (backward compatibility)"""
    img = cv2.imread(image_path)
    result = detect_license_plate_from_image(img)
    return result

@app.route('/detect', methods=['POST'])
def detect():
    """Endpoint nhận diện biển số từ file upload"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'message': 'Không tìm thấy file ảnh'})

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'success': False, 'message': 'Chưa chọn file ảnh'})

        if not allowed_file(image_file.filename):
            return jsonify({'success': False, 'message': 'Định dạng file không được hỗ trợ'})

        filename = secure_filename(image_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(filepath)

        try:
            img = cv2.imread(filepath)
            result = detect_license_plate_from_image(img)

            # Trả về kết quả đầy đủ
            return jsonify(result)

        finally:
            # Xóa file tạm
            if os.path.exists(filepath):
                os.remove(filepath)

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Lỗi server: {str(e)}',
            'plates': [],
            'confidence': 0,
            'processing_time': 0
        })

@app.route('/detect_base64', methods=['POST'])
def detect_base64():
    """Endpoint nhận diện biển số từ base64 image - tối ưu cho real-time"""
    try:
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'message': 'Không tìm thấy dữ liệu ảnh base64',
                'plates': [],
                'confidence': 0
            })

        base64_image = data['image']
        camera_type = data.get('type', 'unknown')  # 'entry' hoặc 'exit'

        # Chuyển đổi base64 thành OpenCV image
        img = base64_to_image(base64_image)
        if img is None:
            return jsonify({
                'success': False,
                'message': 'Không thể decode ảnh base64',
                'plates': [],
                'confidence': 0
            })

        # Nhận diện biển số
        result = detect_license_plate_from_image(img)

        # Thêm thông tin bổ sung
        result['camera_type'] = camera_type
        result['timestamp'] = int(time.time())
        result['detection_details'] = {
            'total_plates_found': len(result['plates']) if result['plates'] else 0,
            'best_confidence': result['confidence'],
            'detection_method': 'YOLOv5 + OCR',
            'image_processed': True
        }

        # Xác định trạng thái và thêm thông tin chi tiết với ưu tiên plate memory
        if result['success'] and result['plates']:
            result['plate'] = result['plates'][0]  # Lấy biển số đầu tiên (đã được sắp xếp ưu tiên)

            # Kiểm tra xem có từ plate memory không
            has_memory_result = result.get('plate_memory_results') and len(result['plate_memory_results']) > 0

            if has_memory_result:
                # Ưu tiên cao cho plate memory
                memory_info = result['plate_memory_results'][0]
                result['status'] = 'detected_memory'
                result['status_message'] = f'🎯 Nhận diện từ Plate Memory: {result["plate"]} (similarity: {memory_info["similarity"]:.2f}, ưu tiên cao)'
                result['detection_source'] = 'plate_memory'
                result['priority'] = 'high'
            elif result['confidence'] > 60:
                result['status'] = 'detected'
                result['status_message'] = f'🤖 OCR nhận diện: {result["plate"]} (Độ tin cậy: {result["confidence"]:.1f}%)'
                result['detection_source'] = 'ocr'
                result['priority'] = 'normal'
            else:
                result['status'] = 'low_confidence'
                result['status_message'] = f'⚠️  Phát hiện biển số: {result["plate"]} nhưng độ tin cậy thấp ({result["confidence"]:.1f}%)'
                result['detection_source'] = 'ocr'
                result['priority'] = 'low'
        else:
            result['status'] = 'no_plate'
            result['status_message'] = 'Không phát hiện biển số trong ảnh'
            result['detection_source'] = 'none'
            result['priority'] = 'none'

        # Log kết quả chi tiết để debug
        print(f"🎯 [{camera_type.upper()}] {result['status_message']} - Time: {result['processing_time']:.2f}s")
        if result['success'] and result['plates']:
            print(f"📋 Detected plates: {result['plates']}")
            if result.get('plate_memory_results'):
                print(f"🧠 Memory results: {[r['plate_text'] for r in result['plate_memory_results']]}")
        else:
            print(f"❌ No plates detected in {camera_type} camera")

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Lỗi server: {str(e)}',
            'plates': [],
            'confidence': 0,
            'processing_time': 0,
            'status': 'error'
        })

@app.route('/health', methods=['GET'])
def health_check():
    """Kiểm tra trạng thái API"""
    return jsonify({
        'status': 'ok',
        'message': 'API nhận diện biển số đang hoạt động bình thường',
        'models_loaded': True,
        'version': '1.0.0',
        'endpoints': {
            'POST /detect': 'Nhận dạng biển số từ file ảnh upload',
            'POST /detect_base64': 'Nhận dạng biển số từ base64 (real-time)',
            'GET /health': 'Kiểm tra trạng thái API',
            'GET /info': 'Thông tin chi tiết về API'
        }
    }), 200

@app.route('/info', methods=['GET'])
def get_info():
    """Thông tin chi tiết về API"""
    return jsonify({
        'api_name': 'License Plate Recognition API',
        'version': '2.0.0',
        'description': 'API nhận diện biển số xe Việt Nam với ưu tiên Plate Memory',
        'models': {
            'detector': 'LP_detector.pt',
            'ocr': 'LP_ocr.pt',
            'confidence_threshold': 0.60,
            'plate_memory_threshold': 0.5
        },
        'features': [
            '🎯 ƯU TIÊN Plate Memory - Nhận dạng biển số đã gán nhãn trước',
            '🤖 OCR fallback - Sử dụng OCR khi không tìm thấy trong memory',
            '📸 Nhận diện từ file ảnh upload',
            '⚡ Real-time từ base64 (camera, webcam)',
            '🧠 Tự động học và ghi nhớ biển số mới',
            '🚀 Tối ưu hóa tốc độ xử lý',
            '🌐 Hỗ trợ CORS cho web integration',
            '📊 API quản lý Plate Memory'
        ],
        'response_format': {
            'success': 'boolean - Trạng thái xử lý',
            'plates': 'array - Danh sách biển số phát hiện',
            'confidence': 'number - Độ tin cậy (0-100)',
            'processing_time': 'number - Thời gian xử lý (giây)',
            'message': 'string - Thông báo kết quả',
            'plate_memory_results': 'array - Kết quả từ plate memory',
            'detection_methods': 'object - Thống kê phương pháp nhận dạng'
        }
    })

@app.route('/plate_memory/stats', methods=['GET'])
def get_plate_memory_stats():
    """Lấy thống kê plate memory"""
    try:
        stats = plate_memory.get_statistics()
        return jsonify({
            'success': True,
            'stats': stats,
            'message': 'Thống kê plate memory'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Lỗi khi lấy thống kê: {str(e)}'
        })

@app.route('/plate_memory/labeled', methods=['GET'])
def get_labeled_plates():
    """Lấy danh sách biển số đã gán nhãn"""
    try:
        labeled_plates = plate_memory.get_labeled_plates()
        return jsonify({
            'success': True,
            'labeled_plates': labeled_plates,
            'count': len(labeled_plates),
            'message': f'Tìm thấy {len(labeled_plates)} biển số đã gán nhãn'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Lỗi khi lấy danh sách: {str(e)}'
        })

@app.route('/plate_memory/assign', methods=['POST'])
def assign_plate_label():
    """Gán nhãn cho biển số"""
    try:
        data = request.get_json()

        if not data or 'plate_id' not in data or 'plate_text' not in data:
            return jsonify({
                'success': False,
                'message': 'Thiếu thông tin plate_id hoặc plate_text'
            })

        plate_id = data['plate_id']
        plate_text = data['plate_text']

        success = plate_memory.assign_plate_text(plate_id, plate_text)

        if success:
            return jsonify({
                'success': True,
                'message': f'Đã gán nhãn "{plate_text}" cho plate_id: {plate_id[:8]}...'
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Không thể gán nhãn cho plate_id: {plate_id}'
            })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Lỗi khi gán nhãn: {str(e)}'
        })

@app.route('/debug/save_camera_image', methods=['POST'])
def save_camera_image():
    """Debug endpoint - Lưu ảnh từ camera để kiểm tra"""
    try:
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'message': 'Không tìm thấy dữ liệu ảnh base64'
            })

        base64_image = data['image']
        camera_type = data.get('type', 'unknown')

        # Chuyển đổi base64 thành OpenCV image
        img = base64_to_image(base64_image)
        if img is None:
            return jsonify({
                'success': False,
                'message': 'Không thể decode ảnh base64'
            })

        # Lưu ảnh để debug
        import os
        debug_dir = "debug_images"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)

        timestamp = int(time.time())
        filename = f"{debug_dir}/camera_{camera_type}_{timestamp}.jpg"
        cv2.imwrite(filename, img)

        # Thử nhận dạng
        result = detect_license_plate_from_image(img)

        return jsonify({
            'success': True,
            'message': f'Đã lưu ảnh debug: {filename}',
            'image_saved': filename,
            'detection_result': result,
            'image_size': f"{img.shape[1]}x{img.shape[0]}"
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Lỗi debug: {str(e)}'
        })

if __name__ == '__main__':
    try:
        print("=" * 60)
        print("🚗 LICENSE PLATE RECOGNITION API SERVER v2.0")
        print("=" * 60)
        print("🎯 Chức năng: ƯU TIÊN Plate Memory + OCR Fallback")
        print("🤖 AI Models: YOLOv5 Full (tương thích plate memory)")
        print("🧠 Plate Memory: Nhận dạng biển số đã gán nhãn TRƯỚC")
        print("⚡ OCR: Sử dụng khi không tìm thấy trong memory")
        print("✅ Models loaded successfully!")

        # Hiển thị thống kê plate memory
        try:
            stats = plate_memory.get_statistics()
            print(f"📊 Plate Memory: {stats['labeled_plates']}/{stats['total_plates']} đã gán nhãn")
        except:
            print("📊 Plate Memory: Đã khởi tạo")

        print("\n📡 Available endpoints:")
        print("  🔸 POST /detect              - Nhận dạng từ file ảnh upload")
        print("  🔸 POST /detect_base64       - Nhận dạng từ base64 (real-time)")
        print("  🔸 GET  /health              - Kiểm tra trạng thái API")
        print("  🔸 GET  /info                - Thông tin chi tiết API")
        print("  🔸 GET  /plate_memory/stats  - Thống kê plate memory")
        print("  🔸 GET  /plate_memory/labeled - Danh sách biển số đã gán nhãn")
        print("  🔸 POST /plate_memory/assign - Gán nhãn cho biển số")
        print("\n🌐 Server URL: http://localhost:5000")
        print("🔧 Mode: Production (threaded=True)")
        print("=" * 60)

        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)

    except Exception as e:
        print(f"❌ Error starting server: {e}")
        print("💡 Kiểm tra:")
        print("   - Port 5000 có đang được sử dụng?")
        print("   - Các file model có tồn tại?")
        print("   - Dependencies đã được cài đặt đầy đủ?")
