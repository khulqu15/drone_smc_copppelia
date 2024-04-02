function trajectory(block)
    setup(block);
    
    function setup(block)
        block.NumDialogPrms  = 4;
        block.NumInputPorts  = 1;
        block.NumOutputPorts = 6;

        block.InputPort(1).Dimensions = 6;
        block.InputPort(1).DirectFeedthrough = false;

        for i = 1:block.NumOutputPorts
            block.OutputPort(i).Dimensions = 1;
        end
        
        block.SampleTimes = [0.1 0];

        block.RegBlockMethod('PostPropagationSetup', @PostPropagationSetup);
        block.RegBlockMethod('InitializeConditions', @InitializeConditions);
        block.RegBlockMethod('Outputs', @Outputs);
        block.RegBlockMethod('Update', @Update);
        block.RegBlockMethod('SetInputPortSamplingMode', @SetInpPortSamplingMode);
    end

    function PostPropagationSetup(block)
        block.NumDworks = 1;
        
        block.Dwork(1).Name = 't';
        block.Dwork(1).Dimensions = 1;
        block.Dwork(1).DatatypeID = 0; % Double
        block.Dwork(1).Complexity = 'Real';
        block.Dwork(1).UsedAsDiscState = true;
    end

    function InitializeConditions(block)
        block.Dwork(1).Data = 0;
    end

    function Outputs(block)
        t = block.Dwork(1).Data;
        
        base_x = block.DialogPrm(1).Data;
        base_y = block.DialogPrm(2).Data;
        
        z_factor = 0.09;
        radius_factor = 0.09;
        angle_speed = 0.2;
        z_start = 0;
        z_end = 6;
        
        angle = t * angle_speed;
        x = base_x + cos(angle) * radius_factor;
        y = base_y + sin(angle) * radius_factor;
        z_increment = min(t * z_factor, z_end - z_start);
        z = z_start + z_increment - 2;
        
        block.OutputPort(1).Data = x;
        block.OutputPort(2).Data = y;
        block.OutputPort(3).Data = z + 2;
        
        block.OutputPort(4).Data = 0;
        block.OutputPort(5).Data = 0;
        block.OutputPort(6).Data = 0;
    end

    function Update(block)
        block.Dwork(1).Data = block.Dwork(1).Data + block.SampleTimes(1);
    end

    function SetInpPortSamplingMode(block, idx, fd)
        block.InputPort(idx).SamplingMode = fd;
        for i = 1:block.NumOutputPorts
            block.OutputPort(i).SamplingMode = fd;
        end
    end
end
