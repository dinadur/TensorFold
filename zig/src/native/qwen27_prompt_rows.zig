//! Each prompt chunk chooses its width from the streams that can decode beside it.
pub fn choose(b: anytype, current: anytype) usize {
    for (b.owners[0..b.runner.slots]) |owner| if (owner) |stream| {
        if (stream != current and !stream.finished and !stream.paused and !b.fills.contains(stream)) {
            return @min(b.chunk, @as(usize, if (b.runner.model.device.tensorUnits()) 512 else 128));
        }
    };
    return b.chunk;
}

/// Decoded spans keep the runner's decode bound even when a solo prompt can take its whole frame.
pub fn chunk(b: anytype, current: anytype, decoded: bool) usize {
    const rows = choose(b, current);
    return if (decoded) @min(rows, @as(usize, b.runner.decodeRows())) else rows;
}
