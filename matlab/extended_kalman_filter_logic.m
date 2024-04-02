function msfcn_ExtendedKalmanFilter(block)
    setup(block);
    
    function setup(block)
        block.NumDialogPrms = 0;
        
        block.NumInputPorts = 2;
        block.NumOutputPorts = 1;
        
        block.InputPort(1).Dimensions = 6; % Data actual: x, y, z, roll, pitch, yaw
        block.InputPort(2).Dimensions = 6; % Data target: x, y, z, roll, pitch, yaw
        
        block.OutputPort(1).Dimensions = 6; % Estimated state
        
        block.SampleTimes = [0.01 0];
        
        block.SimStateCompliance = 'DefaultSimState';
        
        block.RegBlockMethod('InitializeConditions', @InitializeConditions);
        block.RegBlockMethod('Outputs', @Outputs);
        block.RegBlockMethod('PostPropagationSetup', @DoPostPropSetup);
    end

    function DoPostPropSetup(block)
        block.NumDworks = 2;
        block.Dwork(1).Name = 'x';  % State estimate
        block.Dwork(1).Dimensions = 6;
        block.Dwork(1).DatatypeID = 0; % double
        block.Dwork(1).Complexity = 'Real';
        
        block.Dwork(2).Name = 'P';  % Error covariance
        block.Dwork(2).Dimensions = 36; % 6x6 matrix, stored as vector
        block.Dwork(2).DatatypeID = 0; % double
        block.Dwork(2).Complexity = 'Real';
    end

    function InitializeConditions(block)
        % Initialize the state estimate to zero
        block.Dwork(1).Data = zeros(6, 1);
        
        % Initialize the error covariance to identity
        P_init = eye(6);
        block.Dwork(2).Data = P_init(:); % Store as vector
    end

    function Outputs(block)
        % Extract state estimate and error covariance
        x_est = block.Dwork(1).Data;
        P = reshape(block.Dwork(2).Data, [6, 6]); % Reshape back to matrix
        
        % System dynamics (state transition) model
        % NOTE: This should be customized to match your system's dynamics
        dt = block.SampleTimes(1);
        A = eye(6); % Simple placeholder; should be replaced with actual system model
        
        % Measurement model
        % NOTE: Adapt this to your system
        H = eye(6); % Assuming direct measurement
        
        % Process and measurement noise covariance matrices
        % NOTE: Customize these values
        Q = eye(6) * 0.01; % Process noise
        R = eye(6) * 1;    % Measurement noise
        
        % Prediction step
        x_pred = A * x_est;
        P_pred = A * P * A' + Q;
        
        % Measurement update step
        z = block.InputPort(2).Data; % Target measurements as the observed state
        y = z - H * x_pred; % Measurement residual
        S = H * P_pred * H' + R;
        K = P_pred * H' / S; % Kalman gain
        x_est_new = x_pred + K * y;
        P_new = (eye(6) - K * H) * P_pred;
        
        % Update state estimate and error covariance for next time step
        block.Dwork(1).Data = x_est_new;
        block.Dwork(2).Data = P_new(:);
        
        % Output the estimated state
        block.OutputPort(1).Data = x_est_new;
    end
end
