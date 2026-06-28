import { apiClientV2 } from './clientV2'
import type {
  CustomerQuoteRequestV2,
  QuoteResponseV2,
  QuotePreviewResponseV2,
} from '../types/v2'

export const insuranceApiV2 = {
  /**
   * POST /api/v2/quote
   * Full ML prediction with feature enrichment.
   * Returns QuoteResponseV2 (V1 fields + enriched_features + champion_metadata).
   */
  getQuoteV2: async (data: CustomerQuoteRequestV2): Promise<QuoteResponseV2> => {
    const res = await apiClientV2.post('/quote', data)
    return res.data
  },

  /**
   * POST /api/v2/quote/preview
   * Enrichment-only preview — no ML prediction, no DB write.
   * Returns enriched vehicle/city features for review before submitting.
   */
  previewQuoteV2: async (data: CustomerQuoteRequestV2): Promise<QuotePreviewResponseV2> => {
    const res = await apiClientV2.post('/quote/preview', data)
    return res.data
  },
}
