function distance = Distance( p1,p2 )
%DISTANCE Summary of this function goes here
%   Detailed explanation goes here
    
    p2=cell2mat(p2);
    distance = sqrt(((p2(1)-p1(1))^2)+((p2(2)-p1(2))^2)+((p2(3)-p1(3))^2));

end

