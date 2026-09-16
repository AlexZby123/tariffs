import { NextResponse } from "next/server";
import { isLlmReady, readLocalResult, readOverrides } from "@/lib/local";

export const runtime = "nodejs";

export async function GET() {
  const [result, overrides, llmReady] = await Promise.all([
    readLocalResult(),
    readOverrides(),
    isLlmReady(),
  ]);
  return NextResponse.json({ result, overrides, llmReady });
}
