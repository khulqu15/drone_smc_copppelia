function sliding_mode_control_ga_logic(block)
    setup(block);
    
    function setup(block)
        block.NumDialogPrms  = 0;
        
        block.NumInputPorts  = 3;
        block.NumOutputPorts = 7;
        
        block.InputPort(1).Dimensions = 6; % Actual data: x, y, z, roll, pitch, yaw
        block.InputPort(2).Dimensions = 6; % Target data: x, y, z, roll, pitch, yaw
        block.InputPort(3).Dimensions = 24;

        for i = 1:block.NumOutputPorts
            block.OutputPort(i).Dimensions = 1;
        end
        block.OutputPort(7).Dimensions = 6;
        
        block.SampleTimes = [0.01 0];

        block.RegBlockMethod('Outputs', @Outputs);
        block.RegBlockMethod('SetInputPortSamplingMode', @SetInpPortSamplingMode);
    end

    function Outputs(block)
        actual = block.InputPort(1).Data;
        target = block.InputPort(2).Data;
        gaParams = reshape(block.InputPort(3).Data, [4, 6]); % Reshape to [4 parameters, 6 signals]
        
        control_signals = zeros(6,1);
        errors = zeros(6, 1);
        
        for i = 1:6
            e = target(i) - actual(i); % Error
            errors(i) = e;
            
            k = gaParams(1, i);
            lambda = gaParams(2, i);
            alpha = gaParams(3, i);
            rho = gaParams(4, i);

            s = lambda * e; % Sliding surface
            control_signal = k * sign(s) + alpha * s + rho * e^2;
            control_signals(i) = control_signal;
        end
        
        % Assign control signals dan errors ke output ports
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
