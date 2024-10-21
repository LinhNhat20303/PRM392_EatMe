package com.foodapp.eatme.activity;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.widget.Button;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.OptIn;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ExperimentalGetImage;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import com.foodapp.eatme.R;
import com.foodapp.eatme.clickinterface.SimilarityClassifier;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import org.tensorflow.lite.Interpreter;

import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class FaceRecognitionActivity extends AppCompatActivity {
    private static final String TAG = "FaceRecognitionActivity";
    private FaceDetector detector;
    private Interpreter tfLite;
    private PreviewView previewView;
    private ProcessCameraProvider cameraProvider;
    private CameraSelector cameraSelector;
    private boolean startRecognition = true; // Start recognition immediately
    private int inputSize = 112;
    private float[][] embeddings;
    private String modelFile = "mobile_face_net.tflite";
    private HashMap<String, SimilarityClassifier.Recognition> registeredFaces = new HashMap<>();
    private float distanceThreshold = 1.0f;
    private Button btnBack;
    private Bitmap referenceFace;

    @RequiresApi(api = Build.VERSION_CODES.M)
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_face_recognition);
        btnBack = findViewById(R.id.btnCapture);
        previewView = findViewById(R.id.previewView);
        btnBack.setOnClickListener(v -> finish());

        // Load reference face
        referenceFace = loadReferenceFace();

        // Load TensorFlow Lite model
        try {
            tfLite = new Interpreter(loadModelFile(FaceRecognitionActivity.this, modelFile));
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Initialize Face Detector with high accuracy
        FaceDetectorOptions faceDetectorOptions = new FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                .build();
        detector = FaceDetection.getClient(faceDetectorOptions);

        // Camera permission check
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
        } else {
            cameraBind();  // Start camera if permission is already granted
        }
    }

    private void cameraBind() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get();
                Log.d(TAG, "Camera provider obtained");
                bindPreview(cameraProvider);
            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "Error starting camera: ", e);
                Toast.makeText(this, "Error starting camera", Toast.LENGTH_SHORT).show();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    void bindPreview(@NonNull ProcessCameraProvider cameraProvider) {
        Preview preview = new Preview.Builder().build();
        cameraSelector = new CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_FRONT).build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                .setTargetResolution(new Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();

        Executor executor = Executors.newSingleThreadExecutor();
        imageAnalysis.setAnalyzer(executor, new ImageAnalysis.Analyzer() {
            @OptIn(markerClass = ExperimentalGetImage.class)
            @Override
            public void analyze(@NonNull ImageProxy imageProxy) {
                @SuppressLint("UnsafeExperimentalUsageError")
                Image mediaImage = imageProxy.getImage();
                if (mediaImage != null) {
                    InputImage image = InputImage.fromMediaImage(mediaImage, imageProxy.getImageInfo().getRotationDegrees());
                    detector.process(image)
                            .addOnSuccessListener(faces -> {
                                if (!faces.isEmpty()) {
                                    Face face = faces.get(0);
                                    Bitmap frameBitmap = getBitmapFromImageProxy(imageProxy);
                                    Bitmap croppedFace = cropFaceFromBitmap(frameBitmap, face.getBoundingBox());
                                    Bitmap scaledFace = scaleBitmap(croppedFace, inputSize, inputSize);

                                    if (compareFaces(scaledFace, referenceFace)) {
                                        navigateToMainActivity();
                                    }
                                }
                            })
                            .addOnFailureListener(e -> Log.e(TAG, "Face detection failed", e))
                            .addOnCompleteListener(task -> imageProxy.close());
                }
            }
        });

        try {
            cameraProvider.unbindAll();
            cameraProvider.bindToLifecycle((LifecycleOwner) this, cameraSelector, imageAnalysis, preview);
            Log.d(TAG, "Camera bound to lifecycle");
        } catch (IllegalArgumentException e) {
            Log.e(TAG, "No available camera can be found.", e);
            Toast.makeText(this, "No available camera can be found.", Toast.LENGTH_SHORT).show();
        }
    }

    // Helper methods for face recognition
    private MappedByteBuffer loadModelFile(Context context, String modelFile) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelFile);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.getStartOffset(), fileDescriptor.getDeclaredLength());
    }

    private Bitmap getBitmapFromImageProxy(ImageProxy imageProxy) {
        ImageProxy.PlaneProxy[] planes = imageProxy.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];

        // Y plane
        yBuffer.get(nv21, 0, ySize);

        // VU plane
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, imageProxy.getWidth(), imageProxy.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 100, out);
        byte[] imageBytes = out.toByteArray();

        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    private Bitmap cropFaceFromBitmap(Bitmap source, Rect boundingBox) {
        int x = Math.max(0, (int) boundingBox.left);
        int y = Math.max(0, (int) boundingBox.top);
        int width = Math.min((int) boundingBox.width(), source.getWidth() - x);
        int height = Math.min((int) boundingBox.height(), source.getHeight() - y);

        return Bitmap.createBitmap(source, x, y, width, height);
    }

    private Bitmap scaleBitmap(Bitmap bitmap, int width, int height) {
        return Bitmap.createScaledBitmap(bitmap, width, height, false);
    }

    private Bitmap loadReferenceFace() {
        try (InputStream is = getAssets().open("face.jpg")) {
            return BitmapFactory.decodeStream(is);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    private boolean compareFaces(Bitmap detectedFace, Bitmap referenceFace) {
        // Preprocess the bitmaps to the required input size
        Bitmap scaledDetectedFace = scaleBitmap(detectedFace, inputSize, inputSize);
        Bitmap scaledReferenceFace = scaleBitmap(referenceFace, inputSize, inputSize);

        // Get embeddings for both faces
        float[] detectedFaceEmbeddings = getFaceEmbeddings(scaledDetectedFace);
        float[] referenceFaceEmbeddings = getFaceEmbeddings(scaledReferenceFace);

        // Calculate the Euclidean distance between the embeddings
        float distance = calculateEuclideanDistance(detectedFaceEmbeddings, referenceFaceEmbeddings);

        // Compare the distance with the threshold
        return distance < distanceThreshold;
    }

    private float[] getFaceEmbeddings(Bitmap faceBitmap) {
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(faceBitmap);
        float[][] embeddings = new float[1][512]; // Assuming the model outputs 512-dimensional embeddings
        tfLite.run(byteBuffer, embeddings);
        return embeddings[0];
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[inputSize * inputSize];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                final int val = intValues[pixel++];
                byteBuffer.putFloat(((val >> 16) & 0xFF) / 255.0f);
                byteBuffer.putFloat(((val >> 8) & 0xFF) / 255.0f);
                byteBuffer.putFloat((val & 0xFF) / 255.0f);
            }
        }
        return byteBuffer;
    }

    private float calculateEuclideanDistance(float[] embeddings1, float[] embeddings2) {
        float sum = 0.0f;
        for (int i = 0; i < embeddings1.length; i++) {
            float diff = embeddings1[i] - embeddings2[i];
            sum += diff * diff;
        }
        return (float) Math.sqrt(sum);
    }

    private void navigateToMainActivity() {
        Intent intent = new Intent(FaceRecognitionActivity.this, MainActivity.class);
        finishAffinity();
        startActivity(intent);
    }
}