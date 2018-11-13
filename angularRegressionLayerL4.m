classdef angularRegressionLayerL4 < nnet.layer.RegressionLayer
               
    methods
        function layer = angularRegressionLayerL4(name)
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
            angularError = sqrt(abs(1 - ( dot(Y,T)./sqrt(dot(Y,Y))./sqrt(dot(T,T)) ).^2));
    
            % Take mean over mini-batch
            N = size(Y,4);
            loss = sum(angularError)/N;
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            % Returns the derivatives of the angular loss with respect to the predictions Y

            dLdY = (Y.*dot(Y,T) - T.*dot(Y,Y)) ./ (sqrt(dot(Y,Y))).^3 ./ sqrt(dot(T,T)) ./ sqrt(abs(1 - (dot(Y,T)./sqrt(dot(Y,Y))./sqrt(dot(T,T))).^2));
            %dLdY = dot(Y,T) .* (Y.*dot(Y,T) - T.*dot(Y,Y)) ./ (dot(Y,Y)).^2 ./ dot(T,T) ./ sqrt(1 - (dot(Y,T)./sqrt(dot(Y,Y))./sqrt(dot(T,T))).^2);

        end
    end
end