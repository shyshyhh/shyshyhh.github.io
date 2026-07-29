import { defineCollection, z } from 'astro:content';

const blog = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    description: z.string().optional(),
    date: z.coerce.date(),
    draft: z.boolean().default(false),
    experience: z.enum(['gpt-architecture']).optional(),
    lede: z.string().optional(),
    ogImage: z.string().optional(),
  }),
});

export const collections = { blog };
