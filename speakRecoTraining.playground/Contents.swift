import Cocoa
import CreateML
import CreateMLUI

func vowelTraining() {
    do {
        let data = try MLDataTable(contentsOf: URL(fileURLWithPath: "/Users/verdie_b/Desktop/coreml/vowelTraining/vowelTrainingData-20180823-17h50.json"))
		let seed = Int((Date().timeIntervalSince1970 - Date().timeIntervalSince1970.rounded()) * 1000)
        let (trainingData, testingData) = data.randomSplit(by: 0.90, seed: seed)
        
        //let reco = try MLClassifier(trainingData: trainingData, targetColumn: "vowel")
        let boostedTreeParams = MLBoostedTreeClassifier.ModelParameters(maxIterations: 40)
        let reco = try MLBoostedTreeClassifier(trainingData: trainingData, targetColumn: "vowel", featureColumns: nil, parameters: boostedTreeParams)
        
        let evaluationMetrics = reco.evaluation(on: testingData)
        print("Classification error = \(evaluationMetrics.classificationError)")
        print("Confusion: \(evaluationMetrics.confusion)")
        //
        let metadata = MLModelMetadata(author: "Benoit Verdier", shortDescription: "Vowel reco on face expression", version: "1.0")
        let url = URL(fileURLWithPath: "/Users/verdie_b/Desktop/coreml/vowelTraining/VowelOnFace.mlmodel")
        try reco.write(to: url, metadata: metadata)
    }
    catch {
        print("unable to generate")
    }

}

func wordTraining() {
    
    let builder = MLImageClassifierBuilder()
	
    builder.showInLiveView()
}

vowelTraining()
//wordTraining()
