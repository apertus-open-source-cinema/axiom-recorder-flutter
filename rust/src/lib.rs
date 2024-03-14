use std::{cell::Cell, ffi::c_void, iter::repeat_with, rc::Rc, sync::Arc, time::Duration};

use flume::{bounded, Sender, Receiver};

use irondash_dart_ffi::DartValue;
use irondash_run_loop::RunLoop;
use irondash_texture::SendableTexture;
use irondash_texture::{BoxedPixelData, PayloadProvider, SimplePixelData, Texture};


use std::fs::File;
use std::io::BufWriter;
use std::io::Read;
use std::os::fd::AsRawFd;
use std::path::Path;

use recorder::pipeline_processing::frame::{Rgb, Rgba};
use recorder::pipeline_processing::node::InputProcessingNode;
use recorder::pipeline_processing::node::NodeID;
use recorder::pipeline_processing::processing_context::ProcessingContext;
use recorder::pipeline_processing::processing_graph::ProcessingGraph;
use recorder::pipeline_processing::processing_graph::ProcessingGraphBuilder;
use recorder::pipeline_processing::processing_graph::SerdeNodeConfig;
use recorder::pipeline_processing::puller::pull_ordered;


type Frame = (Vec<u8>, u32, u32);

struct PixelBufferSource {
    pub rx: Receiver<Frame>,
}

impl PayloadProvider<BoxedPixelData> for PixelBufferSource {
    fn get_payload(&self) -> BoxedPixelData {
        let (data, width, height) = self.rx.recv().unwrap();
        SimplePixelData::new_boxed(width as i32, height as i32, data)
    }
}

fn init_on_main_thread(engine_handle: i64) -> irondash_texture::Result<i64> {
    let (tx, rx) = bounded(2);
    let provider = Arc::new(PixelBufferSource { rx });
    let texture = Texture::new_with_provider(engine_handle, provider).unwrap();
    let id = texture.id();
    let texture = texture.into_sendable_texture();

    std::thread::spawn(move || {
        get_images(tx, texture);
    });

    Ok(id)
}

#[no_mangle]
pub extern "C" fn init_texture_example(engine_id: i64, ffi_ptr: *mut c_void, port: i64) {
    irondash_dart_ffi::irondash_init_ffi(ffi_ptr);

    // Schedule initialization on main thread. When completed return the
    // texture id back to dart through a port.
    RunLoop::sender_for_main_thread().unwrap().send(move || {
        let port = irondash_dart_ffi::DartPort::new(port);
        match init_on_main_thread(engine_id) {
            Ok(id) => {
                port.send(id);
            }
            Err(err) => {
                panic!("Error {:?}", err);
                port.send(DartValue::Null);
            }
        }
    });
}

pub fn get_images(tx: Sender<Frame>, texture: Arc<SendableTexture<BoxedPixelData>>) -> anyhow::Result<()> {
    let processing_context = ProcessingContext::default();
    let build_graph = || -> anyhow::Result<(ProcessingGraph, NodeID)> {
        let mut graph_builder = ProcessingGraphBuilder::new();
        graph_builder.add(
            "reader".to_string(),
            serde_yaml::from_str::<SerdeNodeConfig>(&format!(
                "
                            type: WebcamInput
                            device: 8
                            "
            ))?
            .into(),
        )?;
        graph_builder.add(
            "dual_frame_decoder".to_string(),
            serde_yaml::from_str::<SerdeNodeConfig>(
                "
                    type: DualFrameRawDecoder
                    input: <reader
                ",
            )?
            .into(),
        )?;
        graph_builder.add(
            "converter".to_string(),
            serde_yaml::from_str::<SerdeNodeConfig>(
                "
                    type: BitDepthConverter
                    input: <dual_frame_decoder
                ",
            )?
            .into(),
        )?;
        let debayer = graph_builder.add(
            "debayer".to_string(),
            serde_yaml::from_str::<SerdeNodeConfig>(
                "
                    type: DebayerResolutionLoss
                    input: <converter
                ",
            )?
            .into(),
        )?;

        let graph = graph_builder.build(&processing_context)?;

        Ok((graph, debayer))
    };
    let (graph, debayer_id) = build_graph().unwrap();
    let debayer = graph.get_node(debayer_id).assert_input_node().unwrap();


    let caps = debayer.get_caps();
    let rgba_converter = Arc::new(RgbToRgba { input: InputProcessingNode::new(10925.into(), debayer), context: processing_context.clone() });

    let rx = pull_ordered(
        &processing_context,
        0,
        Arc::new(|_| {}),
        InputProcessingNode::new(10924.into(), rgba_converter),
        0,
    );


    let (_, queue) = processing_context.require_vulkan().unwrap();
    while let Ok(oframe) = rx.recv() {
        let frame = oframe.downcast::<(Vec<u8>, u32, u32)>().unwrap();
        std::mem::drop(oframe);
        tx.send(Arc::try_unwrap(frame).unwrap()).unwrap();

        texture.mark_frame_available();
    }


    Ok(())
}


use recorder::pipeline_processing::{
    parametrizable::{Parameterizable, Parameters, ParametersDescriptor},
    payload::Payload,
};


use recorder::pipeline_processing::{
    frame::{FrameInterpretation, Raw},
    node::{Caps, ProcessingNode, Request},
    parametrizable::prelude::*,
};
use async_trait::async_trait;

pub struct RgbToRgba {
    pub input: InputProcessingNode,
    pub context: ProcessingContext,
}

#[async_trait]
impl ProcessingNode for RgbToRgba {
    async fn pull(&self, request: Request) -> anyhow::Result<Payload> {
        let frame = self.input.pull(request).await?;
        let frame = self.context.ensure_cpu_buffer::<Rgb>(&frame).unwrap();

        let interp = Rgba { width: frame.interp.width, height: frame.interp.height, fps: frame.interp.fps };
        let mut rgba_buffer = unsafe {
            let cap = (frame.interp.width * frame.interp.height * 4) as usize;
            let mut vec = Vec::with_capacity(cap);
            vec.set_len(cap);
            vec
        };

        frame.storage.as_slice(|frame| {
            for (src, dst) in frame.chunks_exact(3 * 4).zip(rgba_buffer.chunks_exact_mut(4 * 4)) {
                dst[0..4].copy_from_slice(&src[0..4]);
                dst[4..8].copy_from_slice(&src[3..7]);
                dst[8..12].copy_from_slice(&src[6..10]);
                dst[12..15].copy_from_slice(&src[9..12]);
                dst[3] = 255;
                dst[7] = 255;
                dst[11] = 255;
                dst[15] = 255;
            }
        });

        Ok(Payload::from((rgba_buffer, frame.interp.width as u32, frame.interp.height as u32)))
    }

    fn get_caps(&self) -> Caps { self.input.get_caps() }
}
