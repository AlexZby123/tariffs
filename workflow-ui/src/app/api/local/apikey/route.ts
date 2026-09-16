import { NextResponse } from "next/server";
import { isLlmReady, saveApiKey } from "@/lib/local";

export const runtime = "nodejs";

export async function POST(request: Request) {
  try {
    const { apiKey } = (await request.json()) as { apiKey: string };
    await saveApiKey(apiKey);
    return NextResponse.json({ llmReady: await isLlmReady() });
  } catch (error) {
    return NextResponse.json({ error: String(error) }, { status: 400 });
  }
}
