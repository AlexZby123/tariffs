import { NextResponse } from "next/server";
import {
  readSafeWorkflowConfig,
  saveWorkflowConfig,
  type WorkflowConfig,
} from "@/lib/workflow";

export const runtime = "nodejs";

export async function GET() {
  try {
    return NextResponse.json(await readSafeWorkflowConfig());
  } catch (error) {
    return NextResponse.json(
      { error: `Unable to read configuration: ${String(error)}` },
      { status: 500 },
    );
  }
}

export async function PUT(request: Request) {
  try {
    return NextResponse.json(
      await saveWorkflowConfig((await request.json()) as WorkflowConfig),
    );
  } catch (error) {
    return NextResponse.json(
      { error: `Unable to save configuration: ${String(error)}` },
      { status: 400 },
    );
  }
}
