classdef AOIPacket

    properties
        start_step {mustBeNumeric}
    end

    methods

        % Constructor
        function obj = AOIPacket(current_step)
            obj.start_step = current_step;
        end

        % Return current AoI
        function age = get_age(obj, current_step)
            age = current_step - obj.start_step + 1;  % To prevent 0 AoI
        end

    end
end