/// Seed handling for the Generate panel. The UI exposes a single integer field with
/// a Random/Fixed toggle; entering −1 or 0 (or choosing Random) means "roll a fresh
/// seed each run", which maps to a `nil` seed on the request so the backend picks one.
public enum Seed {
    public static func resolve(_ raw: Int) -> UInt64? {
        raw <= 0 ? nil : UInt64(raw)
    }
}
