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
        correctedControl = smcOutput + estimationError * toleranceFactor;
        
        % Calculate a weighted average between corrected control and target,
        newState = (correctedControl * toleranceFactor) + (targetData * (1 - toleranceFactor));
    
        % Update the drone's state with this new calculated state
        block.OutputPort(1).Data = newState;
    
        % Filename
        filename = 'droneStateLog.csv';
    
        % Check if the file exists; if not, create it and write the header
        if exist(filename, 'file') == 0
            header = {'X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw', 'Actual X', 'Actual Y', 'Actual Z', 'Actual Roll', 'Actual Pitch', 'Actual Yaw', 'SMC X', 'SMC Y', 'SMC Z', 'SMC Roll', 'SMC Pitch', 'SMC Yaw', 'Target X', 'Target Y', 'Target Z', 'Target Roll', 'Target Pitch', 'Target Yaw', 'EKF X', 'EKF Y', 'EKF Z', 'EKF Roll', 'EKF Pitch', 'EKF Yaw'};
            fid = fopen(filename, 'w');
            fprintf(fid, '%s,', header{1:end-1});
            fprintf(fid, '%s\n', header{end});
            fclose(fid);
        end
    
        % Combine all data into a single row for logging
        dataToLog = [newState', actualData', smcOutput', targetData', ekfEstimated'];
    
        % Append the new state data and additional data to the CSV file
        dlmwrite(filename, dataToLog, '-append');
    end

end
