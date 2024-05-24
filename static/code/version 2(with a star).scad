// Container and Lid Dimensions
container_diameter = 100; // Diameter of the container in mm
container_height = 150;   // Height of the container in mm
lid_overhang = 20;        // Overhang of the lid in mm
lid_height = 20;          // Height of the lid in mm
thread_height = 5;        // Height of the screw thread in mm
thread_pitch = 2;         // Pitch of the screw thread in mm

module container() {
    difference() {
        // Main body of the container
        cylinder(d = container_diameter, h = container_height, $fn=100);
        // Hollow out the container
        translate([0, 0, thread_height])
            cylinder(d = container_diameter - 4, h = container_height - thread_height, $fn=100);
    }
}

module lid() {
    difference() {
        // Main body of the lid
        cylinder(d = container_diameter + lid_overhang, h = lid_height, $fn=100);
        // Hollow out the lid
        translate([0, 0, thread_height])
            cylinder(d = container_diameter - 4, h = lid_height - thread_height, $fn=100);
    }
}

module screw_thread() {
    // Placeholder for screw thread implementation
    // Note: OpenSCAD does not natively support threads, consider using a library or custom implementation
    // This is a simplified representation
    translate([0, 0, container_height - thread_height])
        cylinder(d = container_diameter - 2, h = thread_height, $fn=100);
}

module star() {
    // Define the star shape using the polygon function
    polygon(points=[
        [0, 5], [1.18, 1.62], [4.76, 1.62], [1.9, -0.62], [3.09, -4.24],
        [0, -2], [-3.09, -4.24], [-1.9, -0.62], [-4.76, 1.62], [-1.18, 1.62]
    ]);
}

module model() {
    container();
    translate([0, 0, container_height])
        lid();
    screw_thread();
    // Add the star on the side of the container
    translate([container_diameter / 2 + 1, 0, container_height / 2])
        rotate([0, 90, 0])
        star();
}

model();