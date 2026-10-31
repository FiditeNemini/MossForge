import MLX

/// Proves the Metal runtime evaluates: sum of [1…5] doubled = 30.
public enum MLXSmoke {
    public static func run() -> Float {
        let x = MLXArray([1, 2, 3, 4, 5] as [Float]) * 2
        return x.sum().item(Float.self)
    }
}
