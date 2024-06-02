function optim_generic_algorithm_quadcopter(block)
    setup(block);
    
    function setup(block)
        block.NumDialogPrms = 0;
        block.NumInputPorts = 4;
        block.NumOutputPorts = 1;
        
        block.InputPort(1).Dimensions = 6;
        block.InputPort(2).Dimensions = 6;
        block.InputPort(3).Dimensions = 6;
        block.InputPort(4).Dimensions = 6;
        
        block.OutputPort(1).Dimensions = 6;
        block.SampleTimes = [0.01 0];
        block.SimStateCompliance = 'DefaultSimState';
        
        block.RegBlockMethod('Outputs', @Outputs);
    end

    function Outputs(block)
        smcOutput = block.InputPort(1).Data;
        actualData = block.InputPort(2).Data;
        targetData = block.InputPort(3).Data;
        ekfEstimated = block.InputPort(4).Data;

        optimizedParams = optimizeParams(targetData, actualData, smcOutput, ekfEstimated);
        optimizedState = calculateNewState(optimizedParams, actualData, smcOutput, ekfEstimated);
        block.OutputPort(1).Data = optimizedState;
    end

    function params = optimizeParams(target, actual, smc, ekf)
        objective = @(x) objectiveWrapper(x, target, actual, smc, ekf);
        nvars = 6;
        lb = zeros(1, nvars);
        ub = ones(1, nvars);
        options = optimoptions('ga', 'Display', 'iter', 'UseParallel', true, 'PopulationSize', 20, 'MaxGenerations', 50);
        [params, ~] = ga(objective, nvars, [], [], [], [], lb, ub, [], options);
    end
    
    function error = objectiveWrapper(params, target, actual, smc, ekf)
        try
            params = reshape(params, [6, 1]);
            newState = calculateNewState(params, actual, smc, ekf);
            error = sum((newState - target).^2, 'all');
        catch
            error = inf;
        end
    end
    
    function newState = calculateNewState(params, actual, smc, ekf)
        params = reshape(params, [6, 1]);
        correctedControl = smc .* params + ekf .* (1 - params);
        newState = reshape(correctedControl, [6, 1]);
    end
end
