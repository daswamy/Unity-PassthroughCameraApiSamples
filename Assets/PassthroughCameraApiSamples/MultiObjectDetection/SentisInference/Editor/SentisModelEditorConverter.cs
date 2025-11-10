// Copyright (c) Meta Platforms, Inc. and affiliates.

using Meta.XR.Samples;
using Unity.Sentis;
using UnityEditor;
using UnityEngine;
using FF = Unity.Sentis.Functional;

namespace PassthroughCameraSamples.MultiObjectDetection.Editor
{
    [MetaCodeSample("PassthroughCameraApiSamples-MultiObjectDetection")]
    [CustomEditor(typeof(SentisInferenceRunManager))]
    public class SentisModelEditorConverter : UnityEditor.Editor
    {
        private const string YOLOV9_FILEPATH = "Assets/PassthroughCameraApiSamples/MultiObjectDetection/SentisInference/Model/yolov9sentis.sentis";
        private const string YOLOV8_FILEPATH = "Assets/PassthroughCameraApiSamples/MultiObjectDetection/SentisInference/Model/yolov8n_sentis.sentis";
        private const string YOLOV5_FILEPATH = "Assets/PassthroughCameraApiSamples/MultiObjectDetection/SentisInference/Model/yolov5n_sentis.sentis";
        private const string YOLOV11_FILEPATH = "Assets/PassthroughCameraApiSamples/MultiObjectDetection/SentisInference/Model/yolov11n_sentis.sentis";
        private SentisInferenceRunManager m_targetClass;
        private float m_iouThreshold;
        private float m_scoreThreshold;
        
        private enum YOLOVersion
        {
            V5,   // Format: [bbox(4), objectness(1), classes(80)] = 85 features
            V8,   // Format: [bbox(4), classes(80)] = 84 features
            V9,   // Format: [bbox(4), confidence(1), classes(80)] = 85 features
            V11   // Format: [bbox(4), classes(80)] = 84 features (same as V8)
        }

        public void OnEnable()
        {
            m_targetClass = (SentisInferenceRunManager)target;
            m_iouThreshold = serializedObject.FindProperty("m_iouThreshold").floatValue;
            m_scoreThreshold = serializedObject.FindProperty("m_scoreThreshold").floatValue;
        }

        public override void OnInspectorGUI()
        {
            _ = DrawDefaultInspector();

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Model Conversion", EditorStyles.boldLabel);
            
            if (GUILayout.Button("Generate YOLOv5 Sentis model with NMS layer"))
            {
                OnEnable();
                ConvertModel(YOLOVersion.V5);
            }
            
            if (GUILayout.Button("Generate YOLOv8 Sentis model with NMS layer"))
            {
                OnEnable();
                ConvertModel(YOLOVersion.V8);
            }
            
            if (GUILayout.Button("Generate YOLOv9 Sentis model with NMS layer"))
            {
                OnEnable();
                ConvertModel(YOLOVersion.V9);
            }
            
            if (GUILayout.Button("Generate YOLOv11 Sentis model with NMS layer"))
            {
                OnEnable();
                ConvertModel(YOLOVersion.V11);
            }
        }

        private void ConvertModel(YOLOVersion version)
        {
            if (m_targetClass.OnnxModel == null)
            {
                Debug.LogError("Please assign an ONNX model to the OnnxModel field first!");
                return;
            }

            try
            {
                //Load model
                var model = ModelLoader.Load(m_targetClass.OnnxModel);
                string versionName = version switch
                {
                    YOLOVersion.V5 => "YOLOv5",
                    YOLOVersion.V8 => "YOLOv8",
                    YOLOVersion.V9 => "YOLOv9",
                    YOLOVersion.V11 => "YOLOv11",
                    _ => "YOLO"
                };
                Debug.Log($"Converting {versionName} model...");

                //Here we transform the output of the model by feeding it through a Non-Max-Suppression layer.
                var graph = new FunctionalGraph();
                var input = graph.AddInput(model, 0);

                var centersToCornersData = new[]
                {
                            1,      0,      1,      0,
                            0,      1,      0,      1,
                            -0.5f,  0,      0.5f,   0,
                            0,      -0.5f,  0,      0.5f
                };
                var centersToCorners = FF.Constant(new TensorShape(4, 4), centersToCornersData);
                
                // Get model output - YOLO models typically output (1, N, features) format
                // YOLOv5: (1, N, 85) = [bbox(4), objectness(1), classes(80)]
                // YOLOv8: (1, N, 84) = [bbox(4), classes(80)]
                // YOLOv9: (1, N, 85) = [bbox(4), confidence(1), classes(80)]
                // YOLOv11: (1, N, 84) = [bbox(4), classes(80)] - same as V8
                var modelOutput = FF.Forward(model, input)[0];
                
                // Extract bounding box coordinates (first 4 elements)
                // modelOutput[0, ..4, ..] gets batch 0, features 0-3, all detections
                // Then transpose to get (N, 4) format
                var boxCoords = modelOutput[0, ..4, ..].Transpose(0, 1);
                
                FunctionalTensor allScores;
                if (version == YOLOVersion.V8 || version == YOLOVersion.V11)
                {
                    // YOLOv8 & YOLOv11: class scores start immediately after bbox coords (elements 4-83)
                    // Format: [bbox(4), classes(80)]
                    allScores = modelOutput[0, 4.., ..].Transpose(0, 1);
                }
                else
                {
                    // YOLOv5 & YOLOv9: skip objectness/confidence score (element 4), get class scores (elements 5-84)
                    // Format: [bbox(4), objectness/confidence(1), classes(80)]
                    allScores = modelOutput[0, 5.., ..].Transpose(0, 1);
                }
                
                // Get max score and class ID for each detection
                var scores = FF.ReduceMax(allScores, 1);    //shape=(N) - max class score per detection
                var classIDs = FF.ArgMax(allScores, 1); //shape=(N) - class ID with highest score
                
                // Convert center+size format to corner format for NMS
                var boxCorners = FF.MatMul(boxCoords, centersToCorners);    //shape=(N,4)
                
                // Apply Non-Maximum Suppression
                var indices = FF.NMS(boxCorners, scores, m_iouThreshold, m_scoreThreshold);
                
                // Prepare indices for gathering
                var indices2 = indices.Unsqueeze(-1).BroadcastTo(new[] { 4 });  //shape=(N,4)
                
                // Gather filtered results after NMS
                var labelIDs = FF.Gather(classIDs, 0, indices); //shape=(N) - class IDs after NMS
                var coords = FF.Gather(boxCoords, 0, indices2); //shape=(N,4) - bbox coords after NMS

                // Compile the graph with two outputs: coords and labelIDs
                var modelFinal = graph.Compile(coords, labelIDs);

                //Export the model to Sentis format
                ModelQuantizer.QuantizeWeights(QuantizationType.Uint8, ref modelFinal);
                
                string filepath = version switch
                {
                    YOLOVersion.V5 => YOLOV5_FILEPATH,
                    YOLOVersion.V8 => YOLOV8_FILEPATH,
                    YOLOVersion.V9 => YOLOV9_FILEPATH,
                    YOLOVersion.V11 => YOLOV11_FILEPATH,
                    _ => YOLOV8_FILEPATH
                };
                ModelWriter.Save(filepath, modelFinal);

                Debug.Log($"Successfully converted {versionName} model to: {filepath}");
                
                // refresh assets
                AssetDatabase.Refresh();
            }
            catch (System.Exception e)
            {
                Debug.LogError($"Error converting model: {e.Message}\n{e.StackTrace}");
            }
        }
    }
}
