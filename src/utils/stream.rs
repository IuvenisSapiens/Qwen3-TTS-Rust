//! Stream module placeholder
pub struct TTSStream;
pub struct TTSResult {
    pub audio: Vec<f32>,
    pub text: String,
    pub codes: Vec<i64>,
}
impl Default for TTSResult {
    fn default() -> Self {
        Self {
            audio: vec![],
            text: String::new(),
            codes: vec![],
        }
    }
}
