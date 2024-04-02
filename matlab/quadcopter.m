function msfcn_DroneMovementIntegration(block)
    setup(block);
    
    function setup(block)
        block.NumDialogPrms = 0;
        
        block.NumInputPorts = 4;
        block.NumOutputPorts = 1;
        
        % Input 1: SMC output [x, y, z, roll, pitch, yaw] control signals
        block.InputPort(1).Dimensions = 6;
        % Input 2: Actual data [x, y, z, roll, pitch, yaw]
        block.InputPort(2).Dimensions = 6;
        % Input 3: Target data [x, y, z, roll, pitch, yaw]
        block.InputPort(3).Dimensions = 6;
        % Input 4: EKF Estimated state [x, y, z, roll, pitch, yaw]
        block.InputPort(4).Dimensions = 6;
        
        % Output 1: Updated state for drone movement
        block.OutputPort(1).Dimensions = 6;
        
        block.SampleTimes = [0.01 0];
        
        block.SimStateCompliance = 'DefaultSimState';
        
        block.RegBlockMethod('Outputs', @Outputs);
    end

    function Outputs(block)
        % Get inputs
        smcOutput = block.InputPort(1).Data; % SMC control signals
        actualData = block.InputPort(2).Data; % Actual position and orientation
        targetData = block.InputPort(3).Data; % Desired position and orientation
        ekfEstimated = block.InputPort(4).Data; % EKF state estimation
    
        % Define a small error tolerance factor
        toleranceFactor = 0.05; % This can be adjusted based on system requirements
    
        % Calculate the difference between target and EKF estimated states
        % This represents the error in estimation relative to the desired state
        estimationError = targetData - ekfEstimated;
    
        % Adjust the SMC output based on estimation error
        % The idea is to correct the control signals based on the difference
        % between where we want to be (target) and where we think we are (EKF estimation)
        correctedControl = smcOutput + estimationError * toleranceFactor;
        
        % Calculate a weighted average between corrected control and target,
        % prioritizing the target but allowing for some correction based on control signals
        % This helps ensure the output is close to the target while considering control adjustments
        newState = (correctedControl * toleranceFactor) + (targetData * (1 - toleranceFactor));
    
        % Update the drone's state with this new calculated state
        block.OutputPort(1).Data = newState;
    end

end
