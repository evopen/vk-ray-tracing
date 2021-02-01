#![cfg_attr(debug_assertions, allow(dead_code, unused_imports, unused))]
#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

mod engine;

use engine::Engine;

use anyhow::{bail, Context, Result};
use log::{debug, error, info, trace, warn};

fn init_logger() -> Result<()> {
    let log_file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .append(false)
        .truncate(true)
        .open(format!("{}.log", env!("CARGO_PKG_NAME")))?;

    fern::Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "{}[{}][{}] {}",
                chrono::Local::now().format("[%Y-%m-%d][%H:%M:%S]"),
                record.target(),
                record.level(),
                message
            ))
        })
        .level(log::LevelFilter::Error)
        .level_for(env!("CARGO_CRATE_NAME"), log::LevelFilter::Trace)
        .chain(std::io::stdout())
        .chain(log_file)
        .apply()?;
    Ok(())
}

fn main() -> Result<()> {
    init_logger().unwrap();

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .worker_threads(16)
        .build()
        .unwrap();

    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
        .with_title("Box of Chocolates")
        .with_resizable(true)
        .with_transparent(false)
        .build(&event_loop)
        .unwrap();

    rt.block_on(async {
        let mut engine = Engine::new(&window).unwrap();
        engine.init().unwrap();

        event_loop.run(move |event, _, control_flow| {
            *control_flow = winit::event_loop::ControlFlow::Poll;
            match event {
                winit::event::Event::NewEvents(_) => {}
                winit::event::Event::WindowEvent { window_id, event } => {
                    engine.input(&event);
                    match event {
                        winit::event::WindowEvent::CloseRequested => {
                            *control_flow = winit::event_loop::ControlFlow::Exit;
                        }
                        winit::event::WindowEvent::KeyboardInput {
                            device_id,
                            input,
                            is_synthetic,
                        } => match input {
                            winit::event::KeyboardInput {
                                virtual_keycode: Some(winit::event::VirtualKeyCode::Escape),
                                state: winit::event::ElementState::Pressed,
                                ..
                            } => {
                                *control_flow = winit::event_loop::ControlFlow::Exit;
                            }
                            _ => {}
                        },
                        _ => {}
                    }
                }
                winit::event::Event::MainEventsCleared => {
                    window.request_redraw();
                }
                winit::event::Event::RedrawRequested(_) => {
                    engine.update().unwrap();
                    engine.render().unwrap();
                }
                winit::event::Event::RedrawEventsCleared => {}
                winit::event::Event::LoopDestroyed => {}
                _ => {}
            }
        });
    });

    Ok(())
}
