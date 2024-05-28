dataDir = 'C:\Users\Dell XPS White\Desktop\KNNClassifier\ImgDataset';

folders = dir(dataDir);
folders = folders([folders.isdir] & ~ismember({folders.name}, {'.', '..'}));

newSize = [227, 227];
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
        features = cat(4, features, img);
        labels = [labels; label];
    end
end

[~, ~, labels] = unique(labels);

idx = randperm(size(features, 4));
splitPoint = round(0.7 * length(idx));
trainIdx = idx(1:splitPoint);
testIdx = idx(splitPoint+1:end);

trainFeatures = features(:, :, :, trainIdx);
testFeatures = features(:, :, :, testIdx);
trainLabels = labels(trainIdx);
testLabels = labels(testIdx);
net = alexnet;

layers = net.Layers;
layers(end-2) = fullyConnectedLayer(max(labels));
layers(end) = classificationLayer;

options = trainingOptions('sgdm', ...
    'MaxEpochs', 5, ...
    'InitialLearnRate', 1e-4, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

trainingStartTime = tic;
netTransfer = trainNetwork(trainFeatures, categorical(trainLabels), layers, options);
trainingTime = toc(trainingStartTime);

inferenceStartTime = tic;
predictedLabels = classify(netTransfer, testFeatures);
inferenceTime = toc(inferenceStartTime);

accuracy = sum(predictedLabels == categorical(testLabels)) / numel(testLabels);
fprintf('Transfer Learning Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Transfer Learning Efficiency(sec): %.2f seconds\n', inferenceTime);
