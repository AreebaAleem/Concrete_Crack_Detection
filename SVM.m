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
        features = [features; double(img(:))'];
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


trainFeatures = trainFeatures / 255;
testFeatures = testFeatures / 255;

numComponents = 100;
[coeff, trainFeatures, ~, ~, explained, ~] = pca(trainFeatures);
trainFeatures = trainFeatures(:, 1:numComponents);
testFeatures = testFeatures * coeff(:, 1:numComponents);

trainingStartTime_SVM = tic;
SVMModel = fitcsvm(trainFeatures, trainLabels);
trainingTime_SVM = toc(trainingStartTime_SVM);

inferenceStartTime_SVM = tic;
predictedLabels_SVM = predict(SVMModel, testFeatures);
inferenceTime_SVM = toc(inferenceStartTime_SVM);

accuracy_SVM = sum(predictedLabels_SVM == testLabels) / length(testLabels);
fprintf('SVM Accuracy: %.2f%%\n', accuracy_SVM * 100);
fprintf('SVM Efficiency(sec): %.2f seconds\n', inferenceTime_SVM);
