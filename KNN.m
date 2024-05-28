dataDir = 'C:\Users\Dell XPS White\Desktop\KNNClassifier\ImgDataset';

folders = dir(dataDir);
folders = folders([folders.isdir] & ~ismember({folders.name}, {'.', '..'}));

newSize = [64, 64];
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
        features = [features; img(:)'];
        labels = [labels; label];
    end
end

[~, ~, labels] = unique(labels);

idx = randperm(size(features, 1));
splitPoint = round(0.7 * length(idx));
trainIdx = idx(1:splitPoint);
testIdx = idx(splitPoint+1:end);

trainFeatures = features(trainIdx, :);
testFeatures = features(testIdx, :);
trainLabels = labels(trainIdx);
testLabels = labels(testIdx);

k = 5;

trainingStartTime_KNN = tic;
mdl_KNN = fitcknn(trainFeatures, trainLabels, 'NumNeighbors', k);
trainingTime_KNN = toc(trainingStartTime_KNN);

inferenceStartTime_KNN = tic;
predictedLabels_KNN = predict(mdl_KNN, testFeatures);
inferenceTime_KNN = toc(inferenceStartTime_KNN);

accuracy = sum(predictedLabels_KNN == testLabels) / length(testLabels);
fprintf('KNN Accuracy: %.2f%%\n', accuracy * 100);

fprintf('KNN Efficiency(sec): %.2f seconds\n', inferenceTime_KNN);
