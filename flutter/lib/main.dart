import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:tflite/tflite.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_ml_model_downloader/firebase_ml_model_downloader.dart';
import 'dart:io'; // Ensure this import for File type

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp();
  runApp(LoanPredictorApp());
}

class LoanPredictorApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Loan Predictor',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: LoanPredictorPage(),
    );
  }
}

class LoanPredictorPage extends StatefulWidget {
  @override
  _LoanPredictorPageState createState() => _LoanPredictorPageState();
}

class _LoanPredictorPageState extends State<LoanPredictorPage> {
  final _formKey = GlobalKey<FormState>();

  final ageController = TextEditingController();
  final incomeController = TextEditingController();
  final homeOwnershipController = TextEditingController();
  final loanIntentController = TextEditingController();
  final loanGradeController = TextEditingController();
  final loanAmountController = TextEditingController();
  final loanPercentIncomeController = TextEditingController();
  final defaultHistoryController = TextEditingController();
  final creditHistoryLengthController = TextEditingController();

  String _predictionResult = "";
  bool _isModelLoaded = false;
  File? _modelFile;

  @override
  void initState() {
    super.initState();
    _downloadModel();
  }

  Future<void> _downloadModel() async {
    final conditions = FirebaseModelDownloadConditions();

    try {
      final customModel = await FirebaseModelDownloader.instance.getModel(
        "loan",
        FirebaseModelDownloadType.localModel,
        conditions,
      );

      setState(() {
        _modelFile = customModel.file;
        _isModelLoaded = true;
      });

      await loadModel();
    } catch (e) {
      print("Failed to download model: $e");
      setState(() {
        _isModelLoaded = false;
      });
    }
  }

  Future<void> loadModel() async {
    if (_modelFile != null) {
      String? res = await Tflite.loadModel(
        model: _modelFile!.path,
      );
      print("Model loaded: $res");
    }
  }

  Future<void> predict() async {
    if (!_isModelLoaded) {
      print("Model not loaded yet.");
      return;
    }

    var input = [
      [
        double.parse(ageController.text),
        double.parse(incomeController.text),
        double.parse(homeOwnershipController.text),
        double.parse(loanIntentController.text),
        double.parse(loanGradeController.text),
        double.parse(loanAmountController.text),
        double.parse(loanPercentIncomeController.text),
        double.parse(defaultHistoryController.text),
        double.parse(creditHistoryLengthController.text),
      ]
    ];

    // Convert input to Uint8List
    var byteBuffer = Float32List.fromList(input[0]).buffer;
    var inputBytes = Uint8List.view(byteBuffer);

    var output = await Tflite.runModelOnBinary(
      binary: inputBytes,
    );

    setState(() {
      _predictionResult = output!.toString();
    });
  }

  @override
  void dispose() {
    ageController.dispose();
    incomeController.dispose();
    homeOwnershipController.dispose();
    loanIntentController.dispose();
    loanGradeController.dispose();
    loanAmountController.dispose();
    loanPercentIncomeController.dispose();
    defaultHistoryController.dispose();
    creditHistoryLengthController.dispose();
    Tflite.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Loan Predictor'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Form(
          key: _formKey,
          child: ListView(
            children: <Widget>[
              TextFormField(
                controller: ageController,
                decoration: InputDecoration(labelText: 'Enter person\'s age'),
                keyboardType: TextInputType.number,
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return 'Please enter age';
                  }
                  return null;
                },
              ),
              TextFormField(
                controller: incomeController,
                decoration: InputDecoration(labelText: 'Enter person\'s income'),
                keyboardType: TextInputType.number,
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return 'Please enter income';
                  }
                  return null;
                },
              ),
              TextFormField(
                controller: homeOwnershipController,
                decoration: InputDecoration(labelText: 'Enter person\'s home ownership (0 - Rented, 1 - Mortgage, 2 - Own, 3 - Other)'),
                keyboardType: TextInputType.number,
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return 'Please enter home ownership';
                  }
                  return null;
                },
              ),
              TextFormField(
                controller: loanIntentController,
                decoration: InputDecoration(labelText: 'Enter loan intent (1 - Debt Consolidation, 2 - Home Improvement, 3 - Medical Expenses, 4 - Other, 5 - Personal)'),
                keyboardType: TextInputType.number,
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return 'Please enter loan intent';
                  }
                  return null;
                },
              ),
              TextFormField(
                controller: loanGradeController,
                decoration: InputDecoration(labelText: 'Enter loan grade (1 - A, 2 - B, 3 - C, 4 - D, 5 - E, 6 - F, 7 - G)'),
                keyboardType: TextInputType.number,
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return 'Please enter loan grade';
                  }
                  return null;
                },
              ),
              TextFormField(
                controller: loanAmountController,
                decoration: InputDecoration(labelText: 'Enter loan amount'),
                keyboardType: TextInputType.number,
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return 'Please enter loan amount';
                  }
                  return null;
                },
              ),
              TextFormField(
                controller: loanPercentIncomeController,
                decoration: InputDecoration(labelText: 'Enter loan percent of income'),
                keyboardType: TextInputType.number,
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return 'Please enter loan percent of income';
                  }
                  return null;
                },
              ),
              TextFormField(
                controller: defaultHistoryController,
                decoration: InputDecoration(labelText: 'Enter person\'s default history (0 - No, 1 - Yes)'),
                keyboardType: TextInputType.number,
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return 'Please enter default history';
                  }
                  return null;
                },
              ),
              TextFormField(
                controller: creditHistoryLengthController,
                decoration: InputDecoration(labelText: 'Enter person\'s credit history length (in years)'),
                keyboardType: TextInputType.number,
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return 'Please enter credit history length';
                  }
                  return null;
                },
              ),
              SizedBox(height: 20),
              ElevatedButton(
                onPressed: () {
                  if (_formKey.currentState!.validate()) {
                    predict();
                  }
                },
                child: Text('Predict'),
              ),
              SizedBox(height: 20),
              Text(
                'Prediction: $_predictionResult',
                style: TextStyle(fontSize: 20),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
