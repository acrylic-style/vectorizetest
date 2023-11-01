

import type {
  VectorizeIndex,
  Fetcher,
  Request,
} from "@cloudflare/workers-types";

import { CloudflareVectorizeStore } from "langchain/vectorstores/cloudflare_vectorize";
import { CloudflareWorkersAIEmbeddings } from "langchain/embeddings/cloudflare_workersai";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { CheerioWebBaseLoader } from 'langchain/document_loaders/web/cheerio'
import { CloudflareWorkersAI } from 'langchain/llms/cloudflare_workersai'
import { OpenAI } from 'langchain/llms/openai'
import {
  loadQAStuffChain,
  loadQAMapReduceChain,
  loadQARefineChain,
} from "langchain/chains";
import { Ai } from '@cloudflare/ai';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';

export const splitArray = <T>(arr: Array<T>, maxLen: number): Array<Array<T>> => Array.from({length: Math.ceil(arr.length / maxLen)}, (_v, i) => arr.slice(i * maxLen, (i + 1) * maxLen));

export interface Env {
  OPENAI_API_KEY: string;
  VECTORIZE_INDEX: VectorizeIndex;
  KV: KVNamespace;
  AI: Fetcher;
}

export default {
  async fetch(request: Request, env: Env) {
    const { pathname, searchParams } = new URL(request.url);
    const embeddings = new OpenAIEmbeddings({
      openAIApiKey: env.OPENAI_API_KEY,
      modelName: 'text-embedding-ada-002',
    })
    // const embeddings = new CloudflareWorkersAIEmbeddings({
    //   binding: env.AI,
    //   modelName: "@cf/baai/bge-large-en-v1.5",
    // });
    const store = new CloudflareVectorizeStore(embeddings, {
      index: env.VECTORIZE_INDEX,
    });
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 450,
      chunkOverlap: 100,
    })
    if (pathname === "/") {
      const query = searchParams.get('query')!
      const results = await store.similaritySearch(query, 5);
      return Response.json(results);
    } else if (pathname === '/ask') {
      try {
      const query = searchParams.get('query')!
      const results = await store.similaritySearch(query, 12)
      console.log(results.map(d => d.pageContent))
      //const chain = loadQAStuffChain(new CloudflareWorkersAI({ model: '@cf/baai/bge-small-en-v1.5' }))
      const chain = loadQAStuffChain(new OpenAI({ openAIApiKey: env.OPENAI_API_KEY, modelName: 'gpt-3.5-turbo' }))
      return Response.json(await chain.call({
        input_documents: results,
        question: query,
      }))
      } catch (e: any) {
        console.error(e.stack || e)
      }
    } else if (pathname === '/insert' && request.method === 'POST') {
      const body: any = await request.json()
      const splitted = await textSplitter.transformDocuments(body)
      splitted.forEach(d => {
        // d.metadata = {}
        d.metadata.loc = JSON.stringify(d.metadata.loc)
      })
      console.log(`Inserting ${splitted.length} entries`)
      const ids = await store.addDocuments(splitted)
      const storedIds: string[] = await env.KV.get('vectorize-ids', 'json') || []
      await env.KV.put('vectorize-ids', JSON.stringify([...storedIds, ...ids]))
      return Response.json({ success: true })
    } else if (pathname === "/load") {
      const article = searchParams.get('article') || 'Cloudflare'
      try {
        const url = searchParams.get('url') || ('https://en.wikipedia.org/wiki/' + encodeURI(article))
        console.log('Loading ' + url)
        const loader = new CheerioWebBaseLoader(url)
        const docs = await loader.loadAndSplit(textSplitter)
        docs.forEach(d => {
          d.metadata.loc = JSON.stringify(d.metadata.loc)
        })
        console.log('Adding ' + docs.length + ' entries')
        const ids = await store.addDocuments(docs)
        const storedIds: string[] = await env.KV.get('vectorize-ids', 'json') || []
        await env.KV.put('vectorize-ids', JSON.stringify([...storedIds, ...ids]))
      } catch (e: any) {
        console.error(e.stack || e)
        throw e
      }

      return Response.json({ success: true })
    } else if (pathname === "/clear") {
      const ids: string[] = await env.KV.get('vectorize-ids', 'json') || []
      for (const array of splitArray(ids, 20)) {
        await store.delete({ ids: array })
      }
      await env.KV.delete('vectorize-ids')
      return Response.json({ success: true })
    }

    return Response.json({ error: "Not Found" }, { status: 404 })
  },
}
