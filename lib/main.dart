
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'PPE Detection App',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: const PpeDetectionWidget(),
    );
  }
}

class PpeDetectionWidget extends StatefulWidget {
  const PpeDetectionWidget({Key? key}) : super(key: key);

  @override
  _PpeDetectionWidgetState createState() => _PpeDetectionWidgetState();
}

class _PpeDetectionWidgetState extends State<PpeDetectionWidget> {
  ui.Image? image;
  List<Detection> detections = [];
  late Interpreter interpreter;
  bool isProcessing = false;
  static const int inputSize = 640;
  final List<String> classNames = [
    'Hardhat',
    'Mask',
    'NO-Hardhat',
    'NO-Mask',
    'NO-Safety Vest',
    'Person',
    'Safety Cone',
    'Safety Vest',
    'machinery',
    'vehicle'
  ];
  final ImagePicker _picker = ImagePicker();
  double scale = 1.0;
  double dx = 0.0;
  double dy = 0.0;

  @override
  void initState() {
    super.initState();
    loadModel();
  }

  Future<void> loadModel() async {
    try {
      interpreter = await Interpreter.fromAsset('ppe_float16.tflite');
    } catch (e) {
      print("Error loading model: $e");
    }
  }

  Future<void> pickImage(ImageSource source) async {
    setState(() => isProcessing = true);
    try {
      final XFile? file = await _picker.pickImage(source: source);
      if (file == null) return;
      final Uint8List bytes = await file.readAsBytes();
      await processImage(bytes);
    } catch (e) {
      print("Error picking image: $e");
      setState(() => isProcessing = false);
    }
  }

  Future<void> processImage(Uint8List bytes) async {
    try {
      final codec = await ui.instantiateImageCodec(bytes);
      final frame = await codec.getNextFrame();
      final ui.Image decodedImage = frame.image;

      setState(() {
        image = decodedImage;
        detections = [];
      });

      final result = await letterboxImage(decodedImage, inputSize, inputSize);
      scale = result.scale;
      dx = result.dx;
      dy = result.dy;

      final inputTensor = await imageToTensor(result.image);
      final output = processOutput(decodedImage, inputTensor);

      setState(() {
        detections = output;
        isProcessing = false;
      });
    } catch (e) {
      print("Error processing image: $e");
      setState(() => isProcessing = false);
    }
  }

  Future<LetterboxResult> letterboxImage(
      ui.Image src, int targetW, int targetH) async {
    final double srcW = src.width.toDouble();
    final double srcH = src.height.toDouble();
    final double scale = (targetW / srcW).clamp(0.0, targetH / srcH);
    final double newW = srcW * scale;
    final double newH = srcH * scale;
    final double padW = (targetW - newW) / 2;
    final double padH = (targetH - newH) / 2;

    final recorder = ui.PictureRecorder();
    final canvas = Canvas(recorder);
    canvas.drawRect(Rect.fromLTWH(0, 0, targetW.toDouble(), targetH.toDouble()),
        Paint()..color = Colors.black);
    canvas.drawImageRect(
      src,
      Rect.fromLTWH(0, 0, srcW, srcH),
      Rect.fromLTWH(padW, padH, newW, newH),
      Paint(),
    );
    final picture = recorder.endRecording();
    final img = await picture.toImage(targetW, targetH);
    return LetterboxResult(img, scale, padW, padH);
  }

  Future<List<List<List<List<double>>>>> imageToTensor(ui.Image img) async {
    final ByteData? bytes =
        await img.toByteData(format: ui.ImageByteFormat.rawRgba);
    final Uint8List rgba = bytes!.buffer.asUint8List();
    final tensor = List.generate(
        1,
        (_) => List.generate(inputSize,
            (y) => List.generate(inputSize, (x) => List.filled(3, 0.0))));

    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        final int idx = (y * inputSize + x) * 4;
        final r = rgba[idx] / 255.0;
        final g = rgba[idx + 1] / 255.0;
        final b = rgba[idx + 2] / 255.0;
        tensor[0][y][x][0] = r; // Adjust if model expects BGR
        tensor[0][y][x][1] = g;
        tensor[0][y][x][2] = b;
      }
    }
    return tensor;
  }

  List<Detection> processOutput(ui.Image img, dynamic inputTensor) {
    final outputShape = interpreter.getOutputTensor(0).shape;
    final output =
        List.filled(outputShape[1] * outputShape[2], 0.0).reshape(outputShape);
    interpreter.run(inputTensor, output);

    List<Detection> detections = [];
    for (int i = 0; i < outputShape[2]; i++) {
      final xCenter =
          output[0][0][i] * inputSize; // If normalized, multiply by 640
      final yCenter = output[0][1][i] * inputSize;
      final width = output[0][2][i] * inputSize;
      final height = output[0][3][i] * inputSize;

      double maxScore = 0.0;
      int classId = 0;
      for (int j = 4; j < 14; j++) {
        if (output[0][j][i] > maxScore) {
          maxScore = output[0][j][i];
          classId = j - 4;
        }
      }

      if (maxScore < 0.5) continue;

      double x1 = (xCenter - width / 2 - dx) / scale;
      double y1 = (yCenter - height / 2 - dy) / scale;
      double x2 = (xCenter + width / 2 - dx) / scale;
      double y2 = (yCenter + height / 2 - dy) / scale;

      x1 = x1.clamp(0.0, img.width.toDouble());
      y1 = y1.clamp(0.0, img.height.toDouble());
      x2 = x2.clamp(0.0, img.width.toDouble());
      y2 = y2.clamp(0.0, img.height.toDouble());

      detections.add(Detection(
        x1: x1,
        y1: y1,
        x2: x2,
        y2: y2,
        label: classNames[classId],
        confidence: maxScore,
      ));
    }

    nms(detections, 0.45);
    return detections;
  }

  void nms(List<Detection> detections, double threshold) {
    detections.sort((a, b) => b.confidence.compareTo(a.confidence));
    for (int i = 0; i < detections.length; i++) {
      for (int j = i + 1; j < detections.length; j++) {
        if (iou(detections[i], detections[j]) > threshold) {
          detections.removeAt(j);
          j--;
        }
      }
    }
  }

  double iou(Detection a, Detection b) {
    final double x1 = a.x1 > b.x1 ? a.x1 : b.x1;
    final double y1 = a.y1 > b.y1 ? a.y1 : b.y1;
    final double x2 = a.x2 < b.x2 ? a.x2 : b.x2;
    final double y2 = a.y2 < b.y2 ? a.y2 : b.y2;
    final double area = (x2 - x1) * (y2 - y1);
    if (area < 0) return 0.0;
    final double aArea = (a.x2 - a.x1) * (a.y2 - a.y1);
    final double bArea = (b.x2 - b.x1) * (b.y2 - b.y1);
    return area / (aArea + bArea - area);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: _buildAppBar(),
      body: _buildMainContent(),
      floatingActionButton: _buildCameraButton(),
    );
  }

  AppBar? _buildAppBar() {
    return image == null
        ? null
        : AppBar(
            title: const Text('PPE Detection'),
            leading: IconButton(
              icon: const Icon(Icons.arrow_back),
              onPressed: () {
                setState(() {
                  image = null;
                  detections = [];
                });
              },
            ),
          );
  }

  Widget _buildMainContent() {
    if (image != null) {
      return Column(
        children: [
          _buildImageWithDetections(),
          _buildDetectionList(),
        ],
      );
    }
    return _buildWelcomeScreen();
  }

  Widget _buildImageWithDetections() {
    return Expanded(
      flex: 3,
      child: Container(
        padding: const EdgeInsets.all(8),
        constraints: BoxConstraints(
          maxHeight: MediaQuery.of(context).size.height * 0.7,
        ),
        child: Stack(
          children: [
            FittedBox(
              fit: BoxFit.contain,
              child: SizedBox(
                width: image!.width.toDouble(),
                height: image!.height.toDouble(),
                child: CustomPaint(
                  painter: DetectionPainter(
                    image: image!,
                    detections: detections,
                  ),
                ),
              ),
            ),
            if (isProcessing) _buildProcessingOverlay(),
          ],
        ),
      ),
    );
  }

  Widget _buildDetectionList() {
    return Expanded(
      flex: 2,
      child: Container(
        decoration: BoxDecoration(
          color: Colors.grey[200],
          borderRadius: const BorderRadius.vertical(top: Radius.circular(20)),
        ), // Added closing parenthesis and comma here
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.all(12.0),
              child: Text(
                'Detected Items (${detections.length})',
                style: const TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
            Expanded(
              child: Scrollbar(
                child: ListView.builder(
                  padding: const EdgeInsets.only(bottom: 16),
                  itemCount: detections.length,
                  itemBuilder: (context, index) {
                    final det = detections[index];
                    return _buildDetectionItem(det);
                  },
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDetectionItem(Detection det) {
    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      child: ListTile(
        leading: Container(
          width: 24,
          height: 24,
          decoration: BoxDecoration(
            color: _getColorForLabel(det.label),
            shape: BoxShape.circle,
          ),
        ),
        title: Text(
          det.label,
          style: const TextStyle(fontWeight: FontWeight.w500),
        ),
        trailing: Text(
          '${(det.confidence * 100).toStringAsFixed(1)}%',
          style: TextStyle(
            color: Colors.grey[600],
            fontSize: 14,
          ),
        ),
      ),
    );
  }

  Color _getColorForLabel(String label) {
    const colorMap = {
      'NO-Hardhat': Colors.red,
      'NO-Mask': Colors.orange,
      'NO-Safety Vest': Colors.purple,
      'Hardhat': Colors.green,
      'Mask': Colors.blue,
      'Safety Vest': Colors.teal,
    };
    return colorMap[label] ?? Colors.grey;
  }

  Widget _buildProcessingOverlay() {
    return Container(
      color: Colors.black54,
      child: const Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(
              color: Colors.white,
              strokeWidth: 4,
              valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
            ),
            SizedBox(height: 16),
            Text(
              'Analyzing Image...',
              style: TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.w500),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildWelcomeScreen() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Text(
            'PPE Detection App',
            style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 20),
          const Text(
            'Select an image source to start detection',
            style: TextStyle(fontSize: 16),
          ),
          const SizedBox(height: 30),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              _buildSourceButton(Icons.photo_library, 'Gallery',
                  () => pickImage(ImageSource.gallery)),
              const SizedBox(width: 20),
              _buildSourceButton(Icons.camera_alt, 'Camera',
                  () => pickImage(ImageSource.camera)),
            ],
          ),
          if (isProcessing) ...[
            const SizedBox(height: 30),
            const CircularProgressIndicator(),
            const SizedBox(height: 10),
            const Text('Processing image...'),
          ]
        ],
      ),
    );
  }

  Widget _buildSourceButton(
      IconData icon, String text, VoidCallback onPressed) {
    return ElevatedButton.icon(
      icon: Icon(icon),
      label: Text(text),
      onPressed: isProcessing ? null : onPressed,
      style: ElevatedButton.styleFrom(
        padding: const EdgeInsets.symmetric(vertical: 15, horizontal: 20),
      ),
    );
  }

  Widget _buildCameraButton() {
    return FloatingActionButton(
      onPressed: isProcessing ? null : () => pickImage(ImageSource.camera),
      tooltip: 'Take Photo',
      child: const Icon(Icons.camera),
    );
  }

  @override
  void dispose() {
    interpreter.close();
    super.dispose();
  }
}

class LetterboxResult {
  final ui.Image image;
  final double scale;
  final double dx;
  final double dy;

  LetterboxResult(this.image, this.scale, this.dx, this.dy);
}

class Detection {
  final double x1, y1, x2, y2;
  final String label;
  final double confidence;

  Detection(
      {required this.x1,
      required this.y1,
      required this.x2,
      required this.y2,
      required this.label,
      required this.confidence});
}

class DetectionPainter extends CustomPainter {
  final ui.Image image;
  final List<Detection> detections;

  DetectionPainter({required this.image, required this.detections});

  @override
  void paint(Canvas canvas, Size size) {
    canvas.drawImage(image, Offset.zero, Paint());
    for (final det in detections) {
      final rect = Rect.fromLTRB(det.x1, det.y1, det.x2, det.y2);
      final paint = Paint()
        ..color = Colors.red
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2;
      canvas.drawRect(rect, paint);

      TextSpan span = TextSpan(
        text: '${det.label} ${(det.confidence * 100).toStringAsFixed(1)}%',
        style: const TextStyle(color: Colors.red, fontSize: 14),
      );
      TextPainter tp =
          TextPainter(text: span, textDirection: TextDirection.ltr);
      tp.layout();
      tp.paint(canvas, Offset(det.x1, det.y1 - 20));
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
