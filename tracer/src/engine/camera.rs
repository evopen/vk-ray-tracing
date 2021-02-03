use glam::Vec3A as Vec3;

#[derive(Default)]
pub struct Camera {
    position: Vec3,
    front: Vec3,
    yaw: f32,
    pitch: f32,
    world_up: Vec3,
    right: Vec3,
    up: Vec3,
}

impl Camera {
    pub fn new(position: Vec3, look_at: Vec3, world_up: Vec3) -> Self {
        let front: Vec3 = look_at - position;
        let pitch = (front.y / front.length())
            .asin()
            .to_degrees()
            .clamp(-89.0, 89.0);
        let mut yaw = (front.z / front.length()).asin().to_degrees();

        if front.z >= 0.0 && front.x < 0.0 {
            yaw = 180.0 - yaw;
        }

        let mut camera = Self {
            position,
            front,
            yaw,
            pitch,
            world_up,
            ..Default::default()
        };

        camera.update_vectors();

        camera
    }

    fn update_vectors(&mut self) {
        self.front = Vec3::new(
            self.yaw.to_radians().cos() * self.pitch.to_radians().cos(),
            self.pitch.to_radians().sin(),
            self.yaw.to_radians().sin() * self.pitch.to_radians().cos(),
        )
        .normalize();
        self.right = self.front.cross(self.world_up).normalize();
        self.up = self.right.cross(self.front).normalize();
    }
}
