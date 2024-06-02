function particle_swarm_optimization_logic(block)
    setup(block);
    
    function setup(block)
        block.NumDialogPrms = 0;
        
        block.NumInputPorts = 2;
        block.NumOutputPorts = 6;
        
        block.InputPort(1).Dimensions = 6;
        block.InputPort(2).Dimensions = 6;
        
        for i = 1:6
            block.OutputPort(i).Dimensions = 4;
        end
        
        block.SampleTimes = [5, 0];
        
        block.RegBlockMethod('Outputs', @Outputs);
        block.RegBlockMethod('SetInputPortSamplingMode', @SetInpPortSamplingMode);
    end
    
    function Outputs(block)
        actualData = block.InputPort(1).Data;
        targetData = block.InputPort(2).Data;
        
        errorPercentage = calculateErrorPercentage(actualData, targetData);
        
        if errorPercentage > 5
            bounds = repmat([0.1, 2; 0.1, 2; 0.01, 0.5; 1, 20], 6, 1);
            num_particles = 10;
            max_iter = 100;
            
            objectiveFunction = @(x) calculateFitness(x, actualData, targetData);
            
            [optimizedParams, ~] = runPSO(objectiveFunction, bounds, num_particles, max_iter);
            
            for i = 1:6
                block.OutputPort(i).Data = optimizedParams((i-1)*4+1:i*4);
            end
        end
    end
    
    function SetInpPortSamplingMode(block, idx, fd)
        block.InputPort(idx).SamplingMode = fd;
        for i = 1:block.NumOutputPorts
            block.OutputPort(i).SamplingMode = fd;
        end
    end

    function [best_position, best_error] = runPSO(objectiveFunction, bounds, num_particles, max_iter)
        dimension = size(bounds, 1);
        particle_positions = rand(num_particles, dimension) .* (bounds(:, 2)' - bounds(:, 1)') + bounds(:, 1)';
        particle_velocities = zeros(size(particle_positions));
        particle_best_positions = particle_positions;
        particle_best_errors = arrayfun(@(i) objectiveFunction(particle_positions(i, :)), 1:num_particles);
        [best_error, idx] = min(particle_best_errors);
        best_position = particle_positions(idx, :);
    
        % PSO parameters adjustment
        w = 0.9; % Initial inertia weight
        wEnd = 0.4; % Final inertia weight, for more refinement in later iterations
        c1 = 2.5; % Cognitive component - increased for more personal best attraction
        c2 = 1.5; % Social component - decreased for less global best attraction
        phi = c1 + c2;
        K = 2/abs(2 - phi - sqrt(phi^2 - 4*phi)); % Constriction factor
    
        for iter = 1:max_iter
            for i = 1:num_particles
                r1 = rand;
                r2 = rand;
                cognitiveVelocity = c1 * r1 * (particle_best_positions(i, :) - particle_positions(i, :));
                socialVelocity = c2 * r2 * (best_position - particle_positions(i, :));
                particle_velocities(i, :) = K * (w * particle_velocities(i, :) + cognitiveVelocity + socialVelocity);
    
                particle_positions(i, :) = particle_positions(i, :) + particle_velocities(i, :);
                particle_positions(i, :) = max(min(particle_positions(i, :), bounds(:, 2)'), bounds(:, 1)');
    
                current_error = objectiveFunction(particle_positions(i, :));
    
                if current_error < particle_best_errors(i)
                    particle_best_positions(i, :) = particle_positions(i, :);
                    particle_best_errors(i) = current_error;
                end
    
                if current_error < best_error
                    best_position = particle_positions(i, :);
                    best_error = current_error;
                end
            end
    
            w = wEnd + ((w - wEnd) * (max_iter - iter) / max_iter);
        end
    end

    
    function fitness = calculateFitness(params, actualData, targetData)
        params = reshape(params, [4, 6])';
        errors = zeros(6, 1);
        for i = 1:6
            adjustedSignal = adjustSignal(actualData(i), params(i, :));
            errors(i) = (adjustedSignal - targetData(i))^2;
        end
        fitness = 1 / (sum(errors) + 1e-6);
    end
    
    function adjustedSignal = adjustSignal(signal, params)
        k = params(1);
        lambda = params(2);
        delta = params(3);
        rho = params(4);
        adjustedSignal = signal * k + lambda - delta + rho;
    end

    function errorPercentage = calculateErrorPercentage(actualData, targetData)
         errorVector = abs((targetData - actualData) ./ targetData) * 100;
        errorPercentage = mean(errorVector);
    end
end
