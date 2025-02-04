% Load the training and test data
trainData = readtable('train.csv'); % Training data
testData = readtable('test.csv');   % Test data
sampleSub = readtable('sample_submission.csv'); % Submission format (unused for labels)

% Extract labels and features from train data
trainLabelsFull = trainData{:, 1}; % First column is the label
trainImagesFull = trainData{:, 2:end}; % Remaining columns are pixel values

% Split trainData into training and validation sets
cv = cvpartition(height(trainData), 'HoldOut', 0.2); % 80% training, 20% validation
idxTrain = training(cv);
idxVal = test(cv);

% Training data
trainImages = trainImagesFull(idxTrain, :); % Pixel values
trainLabels = trainLabelsFull(idxTrain, :); % Labels

% Validation data
valImages = trainImagesFull(idxVal, :); % Pixel values
valLabels = trainLabelsFull(idxVal, :); % Labels

% Normalize pixel values to [0, 1]
trainImages = double(trainImages) / 255.0;
valImages = double(valImages) / 255.0;

% Reshape the images to 28x28x1 for CNN input
trainImages = reshape(trainImages', 28, 28, 1, []);
valImages = reshape(valImages', 28, 28, 1, []);

% Convert labels to categorical
trainLabels = categorical(trainLabels);
valLabels = categorical(valLabels);

% Debug: Ensure labels contain 10 unique classes
disp('Unique classes in training labels:');
disp(categories(trainLabels));

disp('Unique classes in validation labels:');
disp(categories(valLabels));

% Define the CNN model
layers = [
    imageInputLayer([28 28 1], 'Name', 'input')
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv_1')
    reluLayer('Name', 'relu_1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_1')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv_2')
    reluLayer('Name', 'relu_2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_2')
    fullyConnectedLayer(128, 'Name', 'fc_1')
    reluLayer('Name', 'relu_3')
    dropoutLayer(0.2, 'Name', 'dropout_1')
    fullyConnectedLayer(64, 'Name', 'fc_2')
    reluLayer('Name', 'relu_4')
    dropoutLayer(0.2, 'Name', 'dropout_2')
    fullyConnectedLayer(10, 'Name', 'fc_3') % Output layer for 10 classes (0–9)
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')];

% Training options
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 128, ...
    'ValidationData', {valImages, valLabels}, ...
    'Plots', 'training-progress', ...
    'Verbose', true);

% Train the model
net = trainNetwork(trainImages, trainLabels, layers, options);

% Save the trained model
save('digitRecognizerModel.mat', 'net');
disp('Model saved as digitRecognizerModel.mat');

% Evaluate the model on validation data
predictedLabels = classify(net, valImages);
accuracy = sum(predictedLabels == valLabels) / numel(valLabels);
fprintf('Validation Accuracy: %.2f%%\n', accuracy * 100);

% Save test images for real-time predictions
testImages = testData{:,:}; % Extract test images
testImages = double(testImages) / 255.0; % Normalize
testImages = reshape(testImages', 28, 28, 1, []); % Reshape

% Save test images as .png files
outputFolder = 'TestImages';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end
for i = 1:size(testImages, 4)
    imwrite(squeeze(testImages(:, :, 1, i)), fullfile(outputFolder, sprintf('test_%d.png', i)));
end

disp('Test images saved to TestImages folder.');

