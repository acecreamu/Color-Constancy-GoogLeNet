classdef angularRegressionLayerL3 < nnet.layer.RegressionLayer
               
    methods
        function layer = angularRegressionLayerL3(name)
            % Create an angularRegressionLayer

            % Set layer name
            if nargin == 1
                layer.Name = name;
            end

            % Set layer description
            layer.Description = 'Loss as complemented cosine of an angle between vectors';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % Returns the angular loss between the predictions Y and the training targets T

            % Calculate angular error
            angularError = 1 - ( dot(Y,T)./sqrt(dot(Y,Y))./sqrt(dot(T,T)) ).^2;
    
            % Take mean over mini-batch
            N = size(Y,4);
            loss = sum(angularError)/N;
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            % Returns the derivatives of the angular loss with respect to the predictions Y

            dLdY = 2 * dot(Y,T) .* (Y.*dot(Y,T) - T.*dot(Y,Y)) ./ dot(Y,Y).^2 ./ dot(T,T);
        end
    end
end