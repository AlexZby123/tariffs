import { NextResponse } from "next/server";
import { getRunState, startWorkflow } from "@/lib/workflow";

export const runtime = "nodejs";

export async function GET() {
  return NextResponse.json(getRunState());
}

export async function POST(request: Request) {
  try {
    const options = (await request.json()) as {
      dataset: "to_cluster" | "cla";
      limit: number;
      checkOnly: boolean;
    };
    startWorkflow(options);
    return NextResponse.json(getRunState(), { status: 202 });
  } catch (error) {
    return NextResponse.json({ error: String(error) }, { status: 409 });
  }
}
