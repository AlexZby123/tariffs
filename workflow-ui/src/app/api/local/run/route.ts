import { NextResponse } from "next/server";
import {
  getLocalRunState,
  startLocalRun,
  type LocalRunOptions,
} from "@/lib/local";

export const runtime = "nodejs";

export async function GET() {
  return NextResponse.json(getLocalRunState());
}

export async function POST(request: Request) {
  try {
    startLocalRun((await request.json()) as LocalRunOptions);
    return NextResponse.json(getLocalRunState(), { status: 202 });
  } catch (error) {
    return NextResponse.json({ error: String(error) }, { status: 409 });
  }
}
