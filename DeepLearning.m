dataDir = 'C:\Users\Dell XPS White\Desktop\KNNClassifier\ImgDataset';

folders = dir(dataDir);
folders = folders([folders.isdir] & ~ismember({folders.name}, {'.', '..'}));

newSize = [64, 64];
numClasses = length(folders);

features = [];
labels = [];


for i = 1:length(folders)
    label = folders(i).name;
    imgDir = fullfile(dataDir, label);
    imgs = dir(fullfile(imgDir, '*.jpg'));
    for j = 1:length(imgs)
        imgPath = fullfile(imgDir, imgs(j).name);
        img = imread(imgPath);
        img = imresize(img, newSize);
        img = double(img) / 255;
        features = [features; img(:)']; 
        labels = [labels; i];  
    end
end


idx = randperm(size(features, 1));
features = features(idx, :);
labels = labels(idx, :);


splitPoint = round(0.7 * size(features, 1));
trainFeatures = features(1:splitPoint, :);
testFeatures = features(splitPoint+1:end, :);
trainLabels = labels(1:splitPoint, :);
testLabels = labels(splitPoint+1:end, :);

inputSize = numel(newSize);
hiddenSize = 100; 
outputSize = numClasses;

layers = [
    fullyConnectedLayer(hiddenSize)
    reluLayer
    fullyConnectedLayer(outputSize)
    softmaxLayer
    classificationLayer
];

options = trainingOptions('sgdm', 'MaxEpochs', 5, 'Verbose', true);

trainingStartTime = tic;
net = trainNetwork(trainFeatures', categorical(trainLabels)', layers, options);
trainingTime = toc(trainingStartTime);

inferenceStartTime = tic;
predictedLabels = classify(net, testFeatures');
inferenceTime = toc(inferenceStartTime);

accuracy = sum(predictedLabels == categorical(testLabels)') / numel(testLabels);
fprintf('Deep Learning Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Deep Learning Efficiency(sec): %.2f seconds\n', inferenceTime);
