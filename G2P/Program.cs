using Avalonia;
using Serilog;
using System;

namespace G2P;

class Program {
    [STAThread]
    public static void Main(string[] args) {
        Log.Logger = new LoggerConfiguration()
            .MinimumLevel.Debug()
            .WriteTo.Console(outputTemplate: "[{Timestamp:HH:mm:ss} {Level:u3}] {Message:lj}{NewLine}{Exception}")
            .WriteTo.File(
                path: "logs/g2p-.log",
                rollingInterval: RollingInterval.Day,
                retainedFileCountLimit: 7)
            .CreateLogger();

        try {
            Log.Information("Starting G2P Neural Model Tester application...");
            BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);
        } catch (Exception ex) {
            Log.Fatal(ex, "Application terminated unexpectedly.");
        } finally {
            Log.Information("Shutting down application and flushing logs.");
            Log.CloseAndFlush();
        }
    }

    public static AppBuilder BuildAvaloniaApp() => AppBuilder.Configure < App > ()
        .UsePlatformDetect()
        .WithInterFont()
        .LogToTrace();
}