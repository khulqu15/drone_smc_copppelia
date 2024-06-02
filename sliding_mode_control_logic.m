function sliding_mode_control_logic(block)
    setup(block);
    
    function setup(block)
        block.NumDialogPrms  = 0;
        
        block.NumInputPorts  = 2;
        block.NumOutputPorts = 7;
        
        block.InputPort(1).Dimensions = 6; % Actual data: x, y, z, roll, pitch, yaw
        block.InputPort(2).Dimensions = 6; % Target data: x, y, z, roll, pitch, yaw

        for i = 1:block.NumOutputPorts
            block.OutputPort(i).Dimensions = 1;
        end
        block.OutputPort(7).Dimensions = 6;
        
        block.SampleTimes = [0.01 0];

        block.RegBlockMethod('Outputs', @Outputs);
        block.RegBlockMethod('SetInputPortSamplingMode', @SetInpPortSamplingMode);
    end

    function Outputs(block)
        % Actual and target data
        actual = block.InputPort(1).Data;
        target = block.InputPort(2).Data;
        
        % Control parameters for x, y, z (provided), roll, pitch, yaw (assumed)
        params = struct('x', struct('k', 0.8, 'lambda', 2.0, 'alpha', 0.1, 'rho', 10), ...
                        'y', struct('k', 0.7, 'lambda', 2.0, 'alpha', 0.1, 'rho', 10), ...
                        'z', struct('k', 2.0, 'lambda', 2.0, 'alpha', 0.1, 'rho', 10), ...
                        'roll', struct('k', 1.0, 'lambda', 2.0, 'alpha', 0.1, 'rho', 10), ... % Assumed values
                        'pitch', struct('k', 1.0, 'lambda', 2.0, 'alpha', 0.1, 'rho', 10), ... % Assumed values
                        'yaw', struct('k', 1.0, 'lambda', 2.0, 'alpha', 0.1, 'rho', 10)); % Assumed values
        
        % Calculate control signals for each axis
        control_signals = zeros(6,1);
        errors = zeros(6, 1);
        axis_names = {'x', 'y', 'z', 'roll', 'pitch', 'yaw'};
        for i = 1:6
            axis = axis_names{i};
            e = target(i) - actual(i); % Error
            errors(i) = e;
            s = params.(axis).lambda * e; % Sliding surface
            % Sliding mode control signal calculation
            control_signal = params.(axis).k * sign(s) + params.(axis).alpha * s;
            control_signals(i) = control_signal;
        end
        
        % Assign control signals to output ports
        for i = 1:6
            block.OutputPort(i).Data = control_signals(i);
        end
        block.OutputPort(7).Data = errors;
    end

    function SetInpPortSamplingMode(block, idx, fd)
        block.InputPort(idx).SamplingMode = fd;
        for i = 1:block.NumOutputPorts
            block.OutputPort(i).SamplingMode = fd;
        end
    end
end
