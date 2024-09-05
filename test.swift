import CoreML
import Foundation


var m : MLModel

let url = URL.init(filePath: "/Volumes/DevData/workspace/coreml-stable-diffusion-v1-5/original/compiled/TextEncoder.mlmodelc")

m = try MLModel(contentsOf: url)

var ids : [Int]

ids = [49406,  2242, 11798,  3941,   530,   518,  6267,   267,  1400,
         9977,   267,   949,  3027, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407]
let floatIds = ids.map { Float32($0) }

let inputArray = MLShapedArray<Float32>(scalars: floatIds, shape: [1, 77])

let inputFeatures = try! MLDictionaryFeatureProvider(
  dictionary: ["input_ids": MLMultiArray(inputArray)]
)

let modelOutput = try m.prediction(from: inputFeatures)
var embeddingFeature = modelOutput.featureValue(for: "last_hidden_state")
var y = MLShapedArray<Float32>(converting: embeddingFeature!.multiArrayValue!)

print(y[0][0].scalars)
