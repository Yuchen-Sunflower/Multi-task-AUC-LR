load usps_resampled.mat

IND1 = train_labels(9,:) == 1;           % offset by 1 as we label from zero
IND2 = train_labels(10,:) == 1;
a = train_patterns(:,IND1)'
b = train_patterns(:,IND2)'

train_data = [train_patterns(:,IND1)'; train_patterns(:,IND2)'];
train_label = [ones(sum(IND1),1); 0 * ones(sum(IND2),1)];

ITE1 = test_labels(9,:) == 1;            % offset by 1 as we label from zero
ITE2 = test_labels(10,:) == 1;
c = test_patterns(:,ITE1)'
d = test_patterns(:,ITE2)'

test_data = [test_patterns(:,ITE1)'; test_patterns(:,ITE2)'];
test_label = [ones(sum(ITE1),1); 0 * ones(sum(ITE2),1)];
