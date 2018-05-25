function [labels, centers, distances ] = yourKMeans(feats, k);
%YOURKMEANS k-means algorithm
%
%   Parameters:
%       feats:      n-by-p dimensional features' matrix
%       k:          number of clusters 
%   Output:
%       labels:     the assignment of features to clusters
%       centers:    the clusters' centers
%       distances:  the distances of each data point to each cluster
%

% Turns of a warning which was annoying
    warning('off','all')
    kvalues = [];
    [~,b] = size(feats);

    % Creates random centroids through selecting a random value from the feats. 
    for i = 1:k
        random = randi([1 b],1,1);
        kvalues(:,i) = feats(:,random);
    end
    
    dif = 1;
    threshold = 0.0001;
    SSE = [];
    
    
    %Calculates and updates the labels and centroid positions based on the
    %distance of a value to that centroid.
    while ( dif == 1)
        kvalue2 = kvalues.';
        feats2 = feats.';
        %Calculates the distances
        distances = pdist2(feats2, kvalue2 , 'euclidean');
        %Returns the index of the data point, which is the centroid it is
        %closest to.
        [dist,index] = min(distances,[],2);
        
        
        % Calculates the the mean of the data points in a centroid cluster.
        % If the difference between the current mean and the previous mean
        % for a centroid is larger than 0.0001, continue updating the
        % centroids. Once none exceed the threshold the centroids positions
        % and the clusters have been found.
        dif = 0;
        for i = 1:k
            dif = 0;
            found = find(index == i);
            curr_data = feats(:, found);
            mean1 = mean(curr_data,2);
            dif2 = pdist2(kvalues(:,i).', mean1.' , 'euclidean');
            if dif2 > threshold
                dif = 1;
            end
            %Calculates the distance of a data point to that centroid for
            %each data point in that cluster.
            dis = dist(found);
            md = mean(dis);
            value = sum((md - dis).^2);
            kvalues(:,i) = mean1;
            SSE(:,i) = value;
            
        end

    end
    total =0;
    
    %Calculates the value for the elbow method
    for i= 1:k
        total = total + SSE(:,i);
    end
    %Return the SSE for the elbow method, the centers and the labels
    total;
    centers = kvalues;
    labels = index;
end