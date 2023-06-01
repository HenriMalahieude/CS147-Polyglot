const std = @import("std");

fn stoi(string: []u8) !i32 {
    var numerical: i32 = 0;
    for (0..string.len) |i| {
        var char: i32 = @as(i32, string[i]) - 48;

        if (char > 9) return error.NaN;

        if (char >= 0) {
            numerical = numerical * 10 + char;
        }
    }

    return numerical;
}

pub fn main() !void {
    const stdin = std.io.getStdIn().reader();
    const stdout = std.io.getStdOut().writer();

    const random_number = std.crypto.random.intRangeAtMost(i32, 0, 100);

    var bf: [10]u8 = undefined; //can hold at most 10 characters

    while (true) {
        try stdout.print("Please guess a number between 0 - 100: ", .{});

        if (try stdin.readUntilDelimiterOrEof(bf[0..], '\n')) |input| {
            var numberEntered: i32 = try stoi(input);

            if (numberEntered > random_number) {
                try stdout.print("Your number was too big!\n", .{});
            } else if (numberEntered < random_number) {
                try stdout.print("Your number was too small!\n", .{});
            } else {
                try stdout.print("You got the number!\n", .{});
                break;
            }
        }
    }
}
