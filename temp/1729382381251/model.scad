// bacteriophage by erik wrenholt 2017-02-12
// license: creative commons - attribution

$fn = 12;

leg_count = 6;
leg_width = 1.75;

printable_phage();

module printable_phage() {
    // chop off the bottom of the legs so they are flat on the bottom.
    difference() {
        bacteriophage();
        translate([0,0,-5]) cube([100,100,10], center=true);
    }
}

module bacteriophage() {
    body();
    for(i=[0:leg_count]) {
        rotate((360 / leg_count) * i, [0,0,1])
            leg();
    }
 }

module body() {
    
    // spherical head
    translate([0,0,30]) 
        sphere(10, $fn=12);
    
    // base-plate
    translate([0,0,1.5])
        scale([1,1,0.4])
            rotate(30, [0,0,1])
                sphere(6, $fn=12);

    // helical sheath
    for(i=[2:10]) {
        translate([0,0,i*2])
            scale([1,1,0.5])
                sphere(4);
    }

}

module leg() {
    union() {
        hull() {
            translate([2,0,0]) sphere(leg_width);
            translate([20,0,16]) sphere(leg_width);
        }
        hull() {
            translate([20,0,16]) sphere(leg_width);
            translate([30,0,-4]) sphere(leg_width);
        }
    }
}